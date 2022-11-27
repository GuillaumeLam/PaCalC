from extra.subject_wise_split import subject_wise_split
import extra.GaitLab2Go as GL2G

from load_data import _CACHED_load_surface_data
import util_functions as PaCalC

import argparse
import copy
import numpy as np
import os
import pickle
from random import seed, randint
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import time
import tensorflow as tf

# TODO: 
#	-look into f1 of 1.0 very early, even for test data...? => rounding errors?, data leaking?, WRONG LABEL (no trials => 100%)
# 		-silence PaCalC keras calib warnings

def PaCalC_F1(dtst_seed=214, calib_seed=39, save=False, disable_base_train=False):
	save_path = f'graph/PaCalC(dtst_seed={dtst_seed},calib_seed={calib_seed}).pkl'
	if os.path.exists(save_path):
		return save_path

	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

	ann = make_model(X_tr, Y_tr)

	# train model on X_tr, Y_tr
	if not disable_base_train:
		ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

	#=================
	# Get SW curve
	#=================
	mult_pred = ann.predict(X_te, verbose=0)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	report_dict = classification_report(Y_te, y_hat, target_names=list(range(9)), output_dict=True)
	
	sw_f1_per_label = []
	for i in range(9):
		sw_f1_per_label.append(report_dict[i]['f1-score'])
	print(sw_f1_per_label)
	print(np.mean(sw_f1_per_label))
	#=================

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
		pickle.dump((D, np.array([sw_f1_per_label])), open(save_path,'wb'))

	return save_path

# dtst_cv => multiple dataset subj-split seeds; will calib on diff participant
# calib_cv => multiple calibration rnd-split seeds; will calib on same participants with diff gait cycles
def PaCalC_F1_cv(dtst_cv=4, calib_cv=4, save=False):
	save_path = f'graph/PaCalC(dtst_cv={dtst_cv},calib_cv={calib_cv}).pkl'
	# save_path = 'tmp'
	if os.path.exists(save_path):
		return save_path

	dtst_seeds = [randint(0,1000) for _ in range(0,dtst_cv)]

	out = {}
	sw = []

	for i, dtst_seed in enumerate(dtst_seeds):
		global _cached_Irregular_Surface_Dataset
		_cached_Irregular_Surface_Dataset=None

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

		ann = make_model(X_tr, Y_tr)

		# train model on X_tr, Y_tr
		ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

		#=================
		# Get SW curve
		#=================
		mult_pred = ann.predict(X_te, verbose=0)

		y_hat = np.zeros_like(mult_pred)
		y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

		report_dict = classification_report(Y_te, y_hat, target_names=list(range(9)), output_dict=True)
		
		sw_f1_per_label = []
		for i in range(9):
			sw_f1_per_label.append(report_dict[i]['f1-score'])
		print(sw_f1_per_label)
		print(np.mean(sw_f1_per_label))
		sw.append(sw_f1_per_label)
		#=================

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
		print(f'Seed progress: {i+1}/{dtst_cv}={(i+1)/dtst_cv*100}%')
		print(f'\nDataset Fold Completed for seed:{dtst_cv}\n')
		print('='*30)

	# pad dict entries
	for p_id in out.keys():
		out[p_id] = PaCalC.pad_last_dim(out[p_id])

	print(out)

	if save:
		if not os.path.exists('graph'):
   			os.makedirs('graph')
		pickle.dump((out, sw), open(save_path,'wb'))

	return save_path


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
def main_graph_avg_P(run_loc):
	D,sw = pickle.load(open(run_loc,'rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_general(curves, sw=sw)

def per_label_graph_avg_P(run_loc):
	D,sw = pickle.load(open(run_loc,'rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_per_Y(curves, sw=sw)

def main_graph_indiv_P(run_loc, p_id):
	D = pickle.load(open(run_loc,'rb'))

	if len(D[p_id].shape)==2:
		p_curves = np.array([D[p_id]])
	else:
		p_curves = D[p_id]

	PaCalC.graph_calib_curve_general(p_curves, p_id)

def per_label_graph_indiv_P(run_loc, p_id):
	D = pickle.load(open(run_loc,'rb'))

	if len(D[p_id].shape)==2:
		p_curves = np.array([D[p_id]])
	else:
		p_curves = D[p_id]

	PaCalC.graph_calib_curve_per_Y(p_curves, p_id)

def graph_per_P(run_loc):
	D, sw = pickle.load(open(run_loc,'rb'))

	for p_id, p_curves in D.items():
		print(f'P id: {p_id}')
		PaCalC.graph_calib_curve_general(p_curves, p_id, sw=sw)
		PaCalC.graph_calib_curve_per_Y(p_curves, p_id, sw=sw)

def fast_version():
	single_version()

def med_version():
	high_tier_version(dtst_cv=2, calib_cv=2)

def paper_version():
	high_tier_version(dtst_cv=4, calib_cv=4)

def minimal_base_train_needed():
	single_version(disable_base_train=True) # model can master indiv's with no inital training


def single_version(dtst_seed=214, calib_seed=39):
	s = time.time()
	run_loc = PaCalC_F1(dtst_seed=dtst_seed, calib_seed=calib_seed,save=True)
	e = time.time()

	print('TIME of PaCalC_F1:'+str(e-s)+'s')

	main_graph_avg_P(run_loc)
	per_label_graph_avg_P(run_loc)

	print('Select a P_id:')

	# print(D.keys())

	p_id = 15

	print(f'P id: {p_id}')

	main_graph_indiv_P(run_loc, p_id)
	per_label_graph_indiv_P(run_loc, p_id)

def high_tier_version(dtst_cv=2, calib_cv=2):
	s = time.time()
	run_loc = PaCalC_F1_cv(dtst_cv=dtst_cv, calib_cv=calib_cv, save=True)
	e = time.time()

	print(f'TIME of PaCalC_F1_cv(d-cv={dtst_cv},c-cv={calib_cv}):'+str(e-s)+'s')

	main_graph_avg_P(run_loc)
	per_label_graph_avg_P(run_loc)

	graph_per_P(run_loc)


def test_P_missing_labels():
	P_X, P_Y = np.random.rand(50,100), np.array([[1,0,0,0,0,0,0,0,0]]*25+[[0,1,0,0,0,0,0,0,0]]*25)

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(32, input_dim=100, activation='relu'))
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(P_Y.shape[-1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	matrix = PaCalC.partic_calib_curve(model, P_X, P_Y)

	# print('HERE')
	# print(matrix)

	matrix = np.array([matrix])

	PaCalC.graph_calib_curve_general(matrix)

	PaCalC.graph_calib_curve_per_Y(matrix)

def test_all_missing_labels():
	X, Y, P = np.random.rand(50,100), np.array([[1,0,0,0,0,0,0,0,0]]*25+[[0,1,0,0,0,0,0,0,0]]*25), np.array([1,2,3,4,5]*10)

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(32, input_dim=100, activation='relu'))
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(Y.shape[-1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	d = PaCalC.all_partic_calib_curve(model, X, Y, P)

	# print('HERE')
	# print(matrix)

	# matrix = np.array([matrix])

	matrix = PaCalC.collapse_P(d)

	PaCalC.graph_calib_curve_general(matrix)

	PaCalC.graph_calib_curve_per_Y(matrix)

	for p_id, p_curves in d.items():
		print(f'P id: {p_id}')
		# p_curves = np.array([p_curves])
		PaCalC.graph_calib_curve_general(np.array([p_curves]), p_id)
		PaCalC.graph_calib_curve_per_Y(np.array([p_curves]), p_id)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--version', type=str, help='Which code version to run [fast, med, paper]')
	args = parser.parse_args()

	
	if args.version == 'fast':
		fast_version()				# ~Xmin		1x1
	elif args.version == 'med':		#   V x4	
		med_version()				# ~XXmin	2x2=4
	elif args.version == 'paper':	#   V x4	
		paper_version()				# est. ~1h	4x4=16
	elif args.version == 'test':
		# high_tier_version(dtst_cv=4, calib_cv=1)
		test_all_missing_labels()
	else:
		print('Must select version: `-v [fast, med, paper]`')

	# PaCalC_F1(save=True)
	# print('GREAT SUCCESS !'*5)
	# PaCalC_F1_cv(save=True)
	# print('GREAT SUCCESS !!'*5)

	# main_graph_avg_P()
	# per_label_graph_avg_P()
	# main_graph_indiv_P()
	# per_label_graph_indiv_P()