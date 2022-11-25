from load_data import _CACHED_load_surface_data
from extra.subject_wise_split import subject_wise_split
import extra.GaitLab2Go as GL2G

import numpy as np
from sklearn.metrics import f1_score

import copy
import os
import pickle
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf

import PaCalC

irreg_surfaces_labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

# TODO: 
#	-look into f1 of 1.0 very early, even for test data...? => rounding errors?, data leaking?, WRONG LABEL (no trials => 100%)
# 		-silence PaCalC keras calib warnings

def PaCalC_F1(dtst_seed=214, calib_seed=39, save=False):
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

	ann = make_model(X_tr, Y_tr)

	# train model on X_tr, Y_tr
	ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

	#=================
	D = PaCalC.all_partic_calib_curve(ann, X_te, Y_te, P_te, calib_seed)
	#=================
	# or single participant
	#=================
	# participants_dict = PaCalC.perParticipantDict(X_te, Y_te, P_te)
	# p_id = list(participants_dict.keys())[0]
	# matrix = PaCalC.partic_calib_curve(ann, *participants_dict[p_id], calib_seed)
	#=================

	print(D)
	if save:
		if not os.path.exists('graph'):
   			os.makedirs('graph')
		pickle.dump(D, open(f'graph/PaCalC(dtst_seed={dtst_seed},calib_seed={calib_seed}).pkl','wb'))

# dtst_cv => multiple dataset subj-split seeds; will calib on diff participant
# calib_cv => multiple calibration rnd-split seeds; will calib on same participants with diff gait cycles
def PaCalC_dtst_cv(dtst_cv=4, calib_cv=4, save=False):
	dtst_seeds = [randint(0,1000) for _ in range(0,dtst_cv)]

	out = {}

	for dtst_seed in dtst_seeds:
		global _cached_Irregular_Surface_Dataset
		_cached_Irregular_Surface_Dataset=None

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

		ann = make_model(X_tr, Y_tr)

		# train model on X_tr, Y_tr
		ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

		#=================
		D = PaCalC.all_pcc_cv(ann, X_te, Y_te, P_te, cv=calib_cv)
		#=================
		# or single participant
		#=================
		# participants_dict = PaCalC.perParticipantDict(X_te, Y_te, P_te)
		# p_id = list(participants_dict.keys())[0]
		# matrix = PaCalC.ppc_cv(ann, *participants_dict[p_id], cv=calib_cv)
		#=================

		for p_id in D.keys():
			if p_id not in out:
				out[p_id] = []
			for m in D[p_id]:
				out[p_id].append(m)

		print('='*30)
		print('\nDataset Fold Completed\n')
		print('='*30)

	# pad dict entries
	for p_id in out.keys():
		out[p_id] = PaCalC.pad_last_dim(out[p_id])

	print(out)

	if save:
		if not os.path.exists('graph'):
   			os.makedirs('graph')
		pickle.dump(out, open(f'graph/PaCalC(dtst_cv={dtst_cv},calib_cv={calib_cv}).pkl','wb'))


def make_model(X_tr, Y_tr):
	Lab = GL2G.data_processing()
	hid_layers=(606,303,606) #hidden layers
	model='classification' #problem type
	output= Y_tr.shape[-1] #ouput shape 
	input_shape=X_tr.shape[-1]
	ann=Lab.ANN(hid_layers=hid_layers,model=model,output=output,input_shape=input_shape,activation_hid='relu') # relu in hidden layers
	return ann

# def load_sw_rw(model, cv):
# 	try:
# 		sw, rw = pickle.load(open('graph/sw-rw_F1_per_label.pkl','rb'))
# 	except:
# 		# generate numbers with sw & rw

# 	return sw, rw

# TODO: return of graph to save
def main_graph_all_P():
	# D = pickle.load(open('graph/PaCalC(dtst_seed=214,calib_seed=39).pkl','rb'))
	D = pickle.load(open('graph/PaCalC(dtst_cv=2,calib_cv=2).pkl','rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_general(curves)

def per_label_graph_all_P():
	# D = pickle.load(open('graph/PaCalC(dtst_seed=214,calib_seed=39).pkl','rb'))
	D = pickle.load(open('graph/PaCalC(dtst_cv=2,calib_cv=2).pkl','rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_per_Y(curves)

def main_graph_indiv_P():
	# D = pickle.load(open('graph/PaCalC(dtst_seed=214,calib_seed=39).pkl','rb'))
	D = pickle.load(open('graph/PaCalC(dtst_cv=2,calib_cv=2).pkl','rb'))

	if len(D[15].shape)==2:
		p_curves = np.array([D[15]])
	else:
		p_curves = D[15]

	PaCalC.graph_calib_curve_general(p_curves)

def per_label_graph_indiv_P():
	# D = pickle.load(open('graph/PaCalC(dtst_seed=214,calib_seed=39).pkl','rb'))
	D = pickle.load(open('graph/PaCalC(dtst_cv=2,calib_cv=2).pkl','rb'))

	if len(D[15].shape)==2:
		p_curves = np.array([D[15]])
	else:
		p_curves = D[15]

	PaCalC.graph_calib_curve_per_Y(p_curves)

def graph_per_P():
	D = pickle.load(open('graph/PaCalC(dtst_cv=2,calib_cv=2).pkl','rb'))

	for p_id, p_curves in D.items():
		print(p_id)
		PaCalC.graph_calib_curve_general(p_curves)
		PaCalC.graph_calib_curve_per_Y(p_curves)

if __name__ == "__main__":
	# PaCalC_F1(save=True)
	# print('GREAT SUCCESS !'*5)
	# PaCalC_dtst_cv(save=True)
	# print('GREAT SUCCESS !!'*5)

	main_graph_all_P()
	per_label_graph_all_P()
	main_graph_indiv_P()
	per_label_graph_indiv_P()