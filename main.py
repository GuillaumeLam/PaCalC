import GaitLab2Go as GL2G
from extra.subject_wise_split import subject_wise_split

from load_data import _CACHED_load_surface_data
import util_functions as PaCalC

import argparse
import copy
import numpy as np
import os
import pickle
from random import seed, randint
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import time
import tensorflow as tf

def PaCalC_F1_demo():
	X_tr = np.genfromtxt('demo_dataset/X_tr.csv', delimiter=',')
	Y_tr = np.genfromtxt('demo_dataset/Y_tr.csv', delimiter=',')
	P_tr = np.genfromtxt('demo_dataset/P_tr.csv', delimiter=',')
	X_te = np.genfromtxt('demo_dataset/X_te.csv', delimiter=',')
	Y_te = np.genfromtxt('demo_dataset/Y_te.csv', delimiter=',')
	P_te = np.genfromtxt('demo_dataset/P_te.csv', delimiter=',')

	nn = make_ANN(X_tr, Y_tr)

	nn.fit(X_tr, Y_tr, batch_size=512, epochs=50, validation_split=0.1)

	#=================
	# Get SW curve
	#=================
	mult_pred = nn.predict(X_te, verbose=0)

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
	D = PaCalC.all_partic_calib_curve(nn, X_te, Y_te, P_te)

	return D, np.array([sw_f1_per_label]) # D, sw


def PaCalC_F1(dtst_seed=214, calib_seed=39, save=False, disable_base_train=False):
	save_path = f'graph/PaCalC(dtst_seed={dtst_seed},calib_seed={calib_seed},model={model_type}).pkl'
	if os.path.exists(save_path):
		return pickle.load(open(save_path, 'rb'))

	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset = None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1, consent=consent)

	# # CODE TO GENERATE DEMO DATASET
	# print(P_te)
	# print(np.where(P_te[:]==15)[0].shape) # get row index for P=15

	# p_X_te = X_te[np.where(P_te[:]==15)]
	# p_Y_te = Y_te[np.where(P_te[:]==15)]
	# p_P_te = P_te[np.where(P_te[:]==15)]

	# np.savetxt('demo_dataset/X_te.csv', p_X_te[:75,:] , delimiter=',')
	# np.savetxt('demo_dataset/Y_te.csv', p_Y_te[:75,:], delimiter=',')
	# np.savetxt('demo_dataset/P_te.csv', p_P_te[:75], delimiter=',')

	if model_type == 'ANN':
		nn = make_ANN(X_tr, Y_tr)
	elif model_type == 'CNN':
		nn = make_CNN(X_tr, Y_tr)

	# train model on X_tr, Y_tr
	if not disable_base_train:
		nn.fit(X_tr, Y_tr, batch_size=512, epochs=50, validation_split=0.1)

	#=================
	# Get SW curve
	#=================
	mult_pred = nn.predict(X_te, verbose=0)

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
	D = PaCalC.all_partic_calib_curve(nn, X_te, Y_te, P_te, calib_seed)
	#=================
	# or single participant
	#=================
	# participants_dict = PaCalC.perParticipantDict(X_te, Y_te, P_te)
	# p_id = list(participants_dict.keys())[0]
	# matrix = PaCalC.partic_calib_curve(nn, *participants_dict[p_id], calib_seed)
	#=================

	print(D)
	if save:
		if not os.path.exists('graph'):
			os.makedirs('graph')
		pickle.dump((D, np.array([sw_f1_per_label])), open(save_path, 'wb'))

	return D, np.array([sw_f1_per_label]) # D, sw


# dtst_cv => multiple dataset subj-split seeds; will calib on diff participant
# calib_cv => multiple calibration rnd-split seeds; will calib on same participants with diff gait cycles
def PaCalC_F1_cv(dtst_cv=4, save=False):
	save_path = f'graph/PaCalC(dtst_cv={dtst_cv},model={model_type}).pkl'
	# save_path = 'tmp'
	if os.path.exists(save_path):
		return pickle.load(open(save_path, 'rb'))

	dtst_seeds = [randint(0, 1000) for _ in range(0, dtst_cv)]

	out = {}
	sw = []

	for i, dtst_seed in enumerate(dtst_seeds):
		global _cached_Irregular_Surface_Dataset
		_cached_Irregular_Surface_Dataset = None

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1, consent=consent)

		if model_type == 'ANN':
			nn = make_ANN(X_tr, Y_tr)
		elif model_type == 'CNN':
			nn = make_CNN(X_tr, Y_tr)

		# train model on X_tr, Y_tr
		nn.fit(X_tr, Y_tr, batch_size=512, epochs=50, validation_split=0.1)

		#=================
		# Get SW curve
		#=================
		mult_pred = nn.predict(X_te, verbose=0)

		y_hat = np.zeros_like(mult_pred)
		y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

		report_dict = classification_report(Y_te, y_hat, target_names=list(range(9)), output_dict=True)

		sw_f1_per_label = []
		for j in range(9):
			sw_f1_per_label.append(report_dict[j]['f1-score'])
		print(sw_f1_per_label)
		print(np.mean(sw_f1_per_label))
		sw.append(sw_f1_per_label)
		#=================

		#=================
		D = PaCalC.all_partic_calib_curve(nn, X_te, Y_te, P_te, seed=dtst_seed)
		#=================
		# or single participant
		#=================
		# participants_dict = PaCalC.perParticipantDict(X_te, Y_te, P_te)
		# p_id = list(participants_dict.keys())[0]
		# matrix = PaCalC.ppc_cv(nn, *participants_dict[p_id], cv=calib_cv)
		#=================

		for p_id in D.keys():
			if p_id not in out:
				out[p_id] = []
			for m in D[p_id]:
				out[p_id].append(m)

		print('='*30)
		print(f'Seed progress: {i+1}/{dtst_cv}={(i+1)/dtst_cv*100}%')
		print(f'\nDataset Fold Completed for seed:{dtst_seed}\n')
		print('='*30)

	# pad dict entries
	for p_id in out.keys():
		out[p_id] = PaCalC.pad_last_dim(out[p_id])

	print(out)

	if save:
		if not os.path.exists('graph'):
			os.makedirs('graph')
		pickle.dump((out, sw), open(save_path, 'wb'))

	return out, sw


def make_ANN(X_tr, Y_tr):
	Lab = GL2G.data_processing()
	hid_layers = (606, 303, 606)  #hidden layers
	model = 'classification'  #problem type
	output = Y_tr.shape[-1]  #ouput shape
	input_shape = X_tr.shape[-1]
	ann = Lab.ANN(hid_layers=hid_layers, model=model, output=output, input_shape=input_shape, activation_hid='relu') # relu in hidden layers
	return ann

def make_CNN(X_tr, Y_tr):
	Lab = GL2G.data_processing()
	output = Y_tr.shape[-1]  #ouput shape
	input_shape = X_tr.shape[-1]
	cnn = Lab.CNN_test(input_shape=(input_shape, 1),output_shape=output) # relu in hidden layers
	return cnn


def main_graph_avg_P(D,sw):
	# D, sw = pickle.load(open(run_loc, 'rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_general(curves, sw=sw)


def per_label_graph_avg_P(D,sw):
	# D, sw = pickle.load(open(run_loc, 'rb'))

	curves = PaCalC.collapse_P(D)

	PaCalC.graph_calib_curve_per_Y(curves, sw=sw)


def main_graph_indiv_P(D, p_id):
	# D = pickle.load(open(run_loc, 'rb'))

	if len(D[p_id].shape) == 2:
		p_curves = np.array([D[p_id]])
	else:
		p_curves = D[p_id]

	PaCalC.graph_calib_curve_general(p_curves, p_id)


def per_label_graph_indiv_P(D, p_id):
	# D = pickle.load(open(run_loc, 'rb'))

	if len(D[p_id].shape) == 2:
		p_curves = np.array([D[p_id]])
	else:
		p_curves = D[p_id]

	PaCalC.graph_calib_curve_per_Y(p_curves, p_id)


def graph_per_P(D,sw):
	# D, sw = pickle.load(open(run_loc, 'rb'))

	for p_id, p_curves in D.items():
		print(f'P id: {p_id}')
		PaCalC.graph_calib_curve_general(p_curves, p_id, sw=sw)
		PaCalC.graph_calib_curve_per_Y(p_curves, p_id, sw=sw)

def demo_version():
	print('='*30)
	print('Running demo version which uses minimal dataset in repo')
	print('='*30)
	s = time.time()
	D,sw = PaCalC_F1_demo()
	e = time.time()

	print('TIME of PaCalC_F1:'+str(e-s)+'s')

	main_graph_avg_P(D,sw)
	per_label_graph_avg_P(D,sw)

def fast_version():
	single_version()


def med_version():
	high_tier_version(dtst_cv=2)


def paper_version():
	high_tier_version(dtst_cv=14)


def minimal_base_train_needed():
	single_version(disable_base_train=True)		# model can master indiv's with no inital training


def single_version(dtst_seed=214, calib_seed=39):
	s = time.time()
	D,sw = PaCalC_F1(dtst_seed=dtst_seed, calib_seed=calib_seed, save=consent)
	e = time.time()

	print('TIME of PaCalC_F1:'+str(e-s)+'s')

	main_graph_avg_P(D,sw)
	per_label_graph_avg_P(D,sw)

	print('Select a P_id:')

	# print(D.keys())

	p_id = 15

	print(f'P id: {p_id}')

	main_graph_indiv_P(D, p_id)
	per_label_graph_indiv_P(D, p_id)


def high_tier_version(dtst_cv=2):
	s = time.time()
	out,sw = PaCalC_F1_cv(dtst_cv=dtst_cv, save=consent)
	e = time.time()

	print(f'TIME of PaCalC_F1_cv(d-cv={dtst_cv}):'+str(e-s)+'s')

	main_graph_avg_P(out,sw)
	per_label_graph_avg_P(out,sw)

	graph_per_P(out,sw)


if __name__ == "__main__":

	# read file for consent if it exists
	if os.path.isfile('CONSENT.txt'):
		consent = True
	else:
		consent = False

		print('='*30)
		print('Do we have your consent to write files to your PC? (yes/no)')
		yes_choices = ['yes', 'y']

		if input().lower() in yes_choices:
			print('Saving consent choice')
			with open("CONSENT.txt", "w") as file:
				file.write('CONSENT=TRUE')

			consent = True
		print('='*30)

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--version', type=str, help='Which code version to run [demo, fast, medium, paper]')
	parser.add_argument('-m', '--model_type', type=str, help='Which neural network architecture [ANN, CNN]', default='ANN')
	args = parser.parse_args()

	if args.model_type in ['ANN', 'CNN']:
		model_type = args.model_type
	else:
		print('Must select model architecture: `-m [ANN, CNN]`')
		sys.exit(1)

	if args.version == 'demo':
		demo_version()
	elif args.version == 'fast':
		fast_version()
	elif args.version == 'medium':
		med_version()
	elif args.version == 'paper':
		paper_version()
	else:
		print('Must select version: `-v [fast, med, paper]`')
		sys.exit(1)
