import os.path
from os import path
import pandas as pd
import pickle
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import GaitLab2Go as GL2GO
from extra.subject_wise_split import subject_wise_split

def lda_featuers(x_train,y_train):
	n_components=np.unique(y_train).shape[0]-1
	lda= LDA(n_components=n_components)
	x_train=lda.fit_transform(x_train,y_train)
	return x_train,lda

# measure size of each dataset and account based on max mem. (if you can only cache 2 but want 4 folds, iterrate params faster than cached dataset)
# input for file size (nxdataset-seed) and in ram
def _CACHED_load_surface_data(seed, *args, **kwargs):
	global _cached_Irregular_Surface_Dataset

	# caching mechanism
	if path.exists('dataset/DTST:IRREGSURF_NORM_LDA-cache.pkl'):
		_cached_Irregular_Surface_Dataset = pickle.load(open('dataset/DTST:IRREGSURF_NORM_LDA-cache.pkl','rb'))

		if seed in _cached_Irregular_Surface_Dataset:
			print(f"DATASET CACHE HIT FOR SEED#{seed}")
			X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _cached_Irregular_Surface_Dataset[seed]
			return X_tr, Y_tr, P_tr, X_te, Y_te, P_te
		else:
			print(f"DATASET CACHE MISS FOR SEED#{seed}")
			X_tr, Y_tr, P_tr, X_te, Y_te, P_te = load_surface_data(seed, *args, **kwargs)
	else:
		print('NO DATASET CACHE')
		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = load_surface_data(seed, *args, **kwargs)
		_cached_Irregular_Surface_Dataset = {}	

	_cached_Irregular_Surface_Dataset[seed] = (X_tr, Y_tr, P_tr, X_te, Y_te, P_te)
	if consent:
		pickle.dump(_cached_Irregular_Surface_Dataset,open('dataset/DTST:IRREGSURF_NORM_LDA-cache.pkl','wb'))
		print(f"ADDING SEEDED DATASET-{seed} TO CACHE")

	return X_tr, Y_tr, P_tr, X_te, Y_te, P_te

# code taken from `Generalizability of deep learning models for predicting outdoor irregular walking surfaces`
def load_surface_data(seed, subject_wise, split=0.3):
	dp=pd.read_pickle('dataset/cycles_Normalized.pkl') #cycle1_with_wrist,cycle1_9surface
	Ndata=dp.Ndata
	for i in range(Ndata['Surface'].shape[0]):
		Ndata['Surface'][i]=Ndata['Surface'][i][0]
		Ndata['Subjects'][i]=Ndata['Subjects'][i][0]

	output_parameters=['trunk_Acc','trunk_Gyr','trunk_Mag',
					   'shankR_Acc','shankR_Gyr','shankR_Mag',
					   'shankL_Acc','shankL_Gyr','shankL_Mag',
					   'thighR_Acc','thighR_Gyr','thighR_Mag',
					   'thighL_Acc','thighL_Gyr','thighL_Mag']
	parameters=list(Ndata.keys())
	for i in output_parameters:
		print(f'extracting_{i}_Magnitude')
		idx=np.array(parameters)[np.where(np.char.find(parameters,i)!=-1)[0].astype('int64')]
		Ndata[f'{i}_Val']=np.sqrt(np.array(Ndata[idx[0]])**2+ np.array(Ndata[idx[1]])**2+np.array(Ndata[idx[2]])**2)

	Surface=Ndata['Surface']
	Participant=Ndata['Subjects']
	
	Lab=GL2GO.data_processing()

	del Ndata['Surface']
	del Ndata['Subjects']

	Participant = [int(num.strip('subject_')) for num in Participant]

	AnnData={'X_train':{},'X_test':{},'y_train':{},'y_test':{}}
	lda={}

	for i in Ndata.keys():
		print(f'extracting 8 features from {i}')
		X_tr, Y_tr, X_te, Y_te, P_tr, P_te = subject_wise_split(np.array(Ndata[i]), np.array(Surface), np.array(Participant), subject_wise=subject_wise, split=split, seed=seed)
		
		x_train,lda[f'{i}']=lda_featuers(X_tr,Y_tr)
		AnnData['X_train'][f'{i}']=x_train
		AnnData['X_test'][f'{i}']=lda[f'{i}'].transform(X_te)
		AnnData['y_train'][f'{i}']=Y_tr
		AnnData['y_test'][f'{i}']=Y_te

	extract = np.unique(P_te)
	output_parameters=['trunk_Acc','trunk_Gyr','trunk_Mag',
					   'shankR_Acc','shankR_Gyr','shankR_Mag',
					   'shankL_Acc','shankL_Gyr','shankL_Mag',
					   'thighR_Acc','thighR_Gyr','thighR_Mag',
					   'thighL_Acc','thighL_Gyr','thighL_Mag']
	parameters=list(AnnData['X_train'].keys())
	
	for i in output_parameters:
		print(f'concatenating X,Y and Z features of {i} to extract 8 new features')
		idx=np.array(parameters)[np.where(np.char.find(parameters,i)!=-1)[0].astype('int64')]
		x_train=np.concatenate([AnnData['X_train'][idx[0]],
								AnnData['X_train'][idx[1]],
								AnnData['X_train'][idx[2]]],axis=-1)
		x_test=np.concatenate([AnnData['X_test'][idx[0]],
							   AnnData['X_test'][idx[1]],
							   AnnData['X_test'][idx[2]]],axis=-1)
		#x_train,lda[f'{i}']=lda_featuers(x_train,Surface[train_index])
		AnnData['X_train'][f'{i}']=x_train
		AnnData['X_test'][f'{i}']=x_test
#         AnnData['y_train'][f'{i}']=Surface[train_index]
#         AnnData['y_test'][f'{i}']=Surface[test_index]
		AnnData['y_train'][f'{i}']=Y_tr
		AnnData['y_test'][f'{i}']=Y_te
		del AnnData['X_train'][idx[0]]
		del AnnData['X_train'][idx[1]]
		del AnnData['X_train'][idx[2]]
		del AnnData['X_test'][idx[0]]
		del AnnData['X_test'][idx[1]]
		del AnnData['X_test'][idx[2]]

	parameters=list(AnnData['X_train'].keys())
	output_parameters=['trunk','thighR','shankR','thighL','shankL']
	for i in output_parameters:
		print(f'concatenating all the features {i}')
		idx=np.array(parameters)[np.where(np.char.find(parameters,i)!=-1)[0].astype('int64')]
		x_train=np.concatenate([AnnData['X_train'][idx[0]],
								AnnData['X_train'][idx[1]],
								AnnData['X_train'][idx[2]],
								AnnData['X_train'][idx[3]],
								AnnData['X_train'][idx[4]],
								AnnData['X_train'][idx[5]]],axis=-1)
		x_test=np.concatenate([AnnData['X_test'][idx[0]],
							   AnnData['X_test'][idx[1]],
							   AnnData['X_test'][idx[2]],
							   AnnData['X_test'][idx[3]],
							   AnnData['X_test'][idx[4]],
							   AnnData['X_test'][idx[5]]],axis=-1)
		AnnData['X_train'][f'{i}']=x_train
		AnnData['X_test'][f'{i}']=x_test
		AnnData['y_train'][f'{i}']=Y_tr
		AnnData['y_test'][f'{i}']=Y_te

	AnnData['X_train']['Lower']=np.concatenate([AnnData['X_train']['trunk'],AnnData['X_train']['thighR'],
												AnnData['X_train']['shankR'],AnnData['X_train']['thighL'],
												AnnData['X_train']['shankL']],axis=-1)
	AnnData['X_test']['Lower']=np.concatenate([AnnData['X_test']['trunk'],AnnData['X_test']['thighR'],
												AnnData['X_test']['shankR'],AnnData['X_test']['thighL'],
												AnnData['X_test']['shankL']],axis=-1)
	
	AnnData['y_train']['Lower']=Y_tr
	AnnData['y_test']['Lower']=Y_te

	# X_tr, Y_tr, P_tr, X_te, Y_te, P_te
	KEY = 'Lower'
	return AnnData['X_train'][KEY], Lab.one_hot(AnnData['y_train'][KEY]), P_tr, AnnData['X_test'][KEY], Lab.one_hot(AnnData['y_test'][KEY]), P_te

def load_raw_surface_data(seed, subject_wise, split=0.3):
	X = np.load('dataset/GoIS_X_norm.npy')
	Y = np.load('dataset/GoIS_Y_norm.npy')
	P = np.load('dataset/GoIS_P_norm.npy')

	X_tr, Y_tr, X_te, Y_te, P_tr, P_te = subject_wise_split(X, Y, P, subject_wise=subject_wise, split=split, seed=seed)

	return X_tr, Y_tr, X_te, Y_te, P_tr, P_te