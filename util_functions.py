# Participant Calibration Curve (PaCalC)

import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import seed, randint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

seed(39)
sw_rw_scores_path = 'extra/sw-rw_F1_per_label.pkl'
sw_rw_labels_path = 'extra/Irregular_Surface_labels.pkl'

#======================>
#  Utility Functions  >
#======================>

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# partic_calib_curve: generate F1 vs C_tr curves per label type for single participant
# in:
#	-model
#	-participant features (X)
#	-participant labels	(Y)
# out:
#	-array of F1 vs C_tr per label type; dim:|unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def partic_calib_curve(model, P_X, P_Y, seed=39):
	f1_lim_threshold = 7
	per_label_dict, min_cycles = perLabelDict(P_X, P_Y)		# do stats w/ min_cycles?

	f1_curves_per_label = []

	i = 1

	n_labels = len(per_label_dict.keys())

	nl_counter = 0

	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
	for pos_y, X in per_label_dict.items():

		if X.size == 0:
			# empty list ie. no gait cycles for selected label
			f1_curves_per_label.append([np.nan])
			continue

		Y = [0]*n_labels
		Y[pos_y] = 1
		Y = np.array([Y]*X.shape[0])

		X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.5, random_state=seed)

		f1_curve = []

		# train model on 1..n gait cycles & eval on else
		for i in range(len(X_tr)):
			
			if i > 0:
				model.fit(X_tr[i-1:i], Y_tr[i-1:i], epochs=50, batch_size=1, verbose=0, callbacks=[callback])

			mult_pred = model.predict(X_te, verbose=0)

			y_hat = np.zeros_like(mult_pred)
			y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

			report_dict = classification_report(Y_te, y_hat, target_names=list(range(n_labels)), output_dict=True)

			f1_curve.append(report_dict[pos_y]['f1-score'])

			if len(f1_curve) > f1_lim_threshold and (f1_curve[-f1_lim_threshold:] == np.array([1.0]*f1_lim_threshold)).all():
				print('Maxing F1, skipping to next label')
				break
			else:
				print('Current F1 trend:', f1_curve)
				print(f'Iteration of C_tr completed {i+1}/{len(X_tr)}={(i+1)/len(X_tr)*100}%')

		f1_curves_per_label.append(f1_curve)
		nl_counter += 1
		print('='*30)
		print(f'Iteration of label completed {nl_counter}/{n_labels}={nl_counter/n_labels*100}%')
		print('='*30)

	# add frq of lengths to running counter, check if group of ppl stand out ie. what are the chances that variations are due to error
	f1_matrix = pad_last_dim(f1_curves_per_label)

	return f1_matrix

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# pcc_cv: partic_calib_curve with different seeds
# out:
#	-array of F1 vs C_tr per label type; dim:|cv| x |unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def pcc_cv(model, P_X, P_Y, cv=2):
	seeds = [randint(0, 1000) for _ in range(0, cv)]

	results = []

	for i, s in enumerate(seeds):
		matrix = partic_calib_curve(model, P_X, P_Y, s)
		results.append(matrix)

		print('='*30)
		print(f'Seed progress: {i+1}/{cv}={(i+1)/cv*100}%')
		print(f'\nCalibration Fold Completed for seed:{s}\n')
		print('='*30)

	return pad_last_dim(results)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# partic_calib_curve: generate F1 vs C_tr curves per label type for all participants
# in:
#	-model
#	-dataset features (X)
#	-dataset one hot labels	(Y)
#	-dataset participant id (P)
# out:
#	-dict of F1 vs C_tr per label type per participant; dim:{|unique(P)|} => |unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def all_partic_calib_curve(model, X, Y, P, seed=39):
	model_cpy = keras_base_model(model)
	weight_chkpnt = model.get_weights()

	participants_data = perParticipantDict(X, Y, P)

	participants_curves = {}

	# repeat partic_calib_curve over all participant
	for i, p_id in enumerate(participants_data.keys()):
		model_cpy.set_weights(weight_chkpnt)
		participants_curves[p_id] = partic_calib_curve(model_cpy, *participants_data[p_id], seed)[np.newaxis, ...]

		# print('HERE'*20)
		# print(participants_curves[p_id].shape)
		# print('HERE'*20)

		print('='*30)
		print(f'P progress: {i+1}/{len(participants_data.keys())}={(i+1)/len(participants_data.keys())*100}%')
		print(f'\nCalibration Curve Computed for P:{p_id}\n')
		print('='*30)

	return participants_curves

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# all_pcc_cv: all_partic_calib_curve with different seeds
# out:
#	-dict of F1 vs C_tr per label type per participant; dim:{|unique(P)|} => |cv| x |unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def all_pcc_cv(model, X, Y, P, cv=2):
	seeds = [randint(0, 1000) for _ in range(0, cv)]

	results = {}

	for i, s in enumerate(seeds):
		dictionary = all_partic_calib_curve(model, X, Y, P, s)

		# integrate folds into results dict
		for p_id in dictionary.keys():
			if p_id not in results:
				results[p_id] = []
			results[p_id].append(dictionary[p_id])

		print('='*30)
		print(f'Seed progress: {i+1}/{cv}={(i+1)/cv*100}%')
		print(f'\nCalibration Fold Completed for seed:{seed}\n')
		print('='*30)

	# pad dict entries
	for p_id in results.keys():
		results[p_id] = pad_last_dim(results[p_id])

	return results

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# graph_calib_curve_per_Y: generate detailed graph of F1 vs C_tr per label type
# in:
#	-F1 vs C_tr curves; dim: n x |unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def graph_calib_curve_per_Y(curves, p_id=None, sw=None):

	# get from dataset/labels.npy
	text_labels = pickle.load(open(sw_rw_labels_path, 'rb'))

	if sw is None:
		sw, rw = pickle.load(open(sw_rw_scores_path, 'rb'))

		sw_avg_f1_l, sw_std_f1_l = sw
		rw_avg_f1_l, rw_std_f1_l = rw
	else:
		sw_avg_f1_l, sw_std_f1_l = np.mean(sw, axis=0), np.std(sw, axis=0)

		_, rw = pickle.load(open(sw_rw_scores_path, 'rb'))
		rw_avg_f1_l, rw_std_f1_l = rw

	for i, surface_label in enumerate(text_labels):
		plt.subplot(3, 3, i+1)
		sw_avg_f1, sw_std_f1 = sw_avg_f1_l[i], sw_std_f1_l[i]
		rw_avg_f1, rw_std_f1 = rw_avg_f1_l[i], rw_std_f1_l[i]
		if i == 0:
			standard_F1_Ctr_graph(curves[:, i, :], ((sw_avg_f1, sw_std_f1), (rw_avg_f1, rw_std_f1)), title_label=text_labels[i])
		else:
			standard_F1_Ctr_graph(curves[:, i, :], ((sw_avg_f1, sw_std_f1), (rw_avg_f1, rw_std_f1)), title_label=text_labels[i], sw_rw_labels=False)

		if i == 3:
			plt.ylabel('F1')
		elif i == 7:
			plt.xlabel('Calibration size')

	plt.figlegend()
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])

	if p_id is None:
		plt.suptitle('F1 vs calibration size per surface types')
	else:
		plt.suptitle(f'F1 vs calibration size per surface types for P_id:{p_id}')

	plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# graph_calib_curve_general: generate graph of F1 vs C_tr averaged over label type
# in:
#	-F1 vs C_tr curves; dim: n x |unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def graph_calib_curve_general(curves, p_id=None, sw=None):
	if sw is None:
		sw, rw = pickle.load(open(sw_rw_scores_path, 'rb'))

		sw_avg_f1_l, _ = sw
		rw_avg_f1_l, _ = rw

		sw_avg_f1 = np.mean(sw_avg_f1_l)
		sw_std_f1 = np.std(sw_avg_f1_l)
		rw_avg_f1 = np.mean(rw_avg_f1_l)
		rw_std_f1 = np.std(rw_avg_f1_l)
	else:
		sw = np.mean(sw, axis=0)
		sw_avg_f1 = np.mean(sw)
		sw_std_f1 = np.std(sw)

		_, rw = pickle.load(open(sw_rw_scores_path, 'rb'))

		rw_avg_f1_l, _ = rw
		rw_avg_f1 = np.mean(rw_avg_f1_l)
		rw_std_f1 = np.std(rw_avg_f1_l)

	f1_Ctr_avged_l = np.nanmean(curves, axis=1)

	standard_F1_Ctr_graph(f1_Ctr_avged_l, ((sw_avg_f1, sw_std_f1), (rw_avg_f1, rw_std_f1)))

	plt.ylabel('F1')
	plt.xlabel('Calibration size')

	plt.xscale('symlog')

	plt.legend(loc='lower right')
	if p_id is None:
		plt.title('F1 vs calibration size (lin vs log) averaged over surface types')
	else:
		plt.title(f'F1 vs calibration size (lin vs log) averaged over surface types for P_id:{p_id}')

	plt.show()

#====================>
#  Helper Functions  >
#====================>


# perLabelDict: make dict of gait cycles per label of participant
# in:
# 	-one hot labels
# return:
# 	-dict of gait cycles per label,
# 	-min number of gait cycles of all labels
def perLabelDict(P_X, P_Y):
	label_dict = {}

	for i in range(P_Y.shape[-1]):
		label_dict[i] = []

	for i, OHE_y in enumerate(P_Y):
		pos_y = np.array(OHE_y).argmax()

		label_dict[pos_y].append(P_X[i])

	for k in label_dict:
		P_X = label_dict[k]
		label_dict[k] = np.array(P_X)

	min_cycles = sys.maxsize

	for i in range(0, len(label_dict.keys())):
		if min_cycles > np.array(label_dict[i]).shape[0]:
			min_cycles = np.array(label_dict[i]).shape[0]

	return label_dict, min_cycles


# arr => array of array
def pad_last_dim(arr):
	# find longest length sub array
	l = 0
	for sub in arr:
		l_sub = np.array(sub).shape[-1]
		if l_sub > l:
			l = l_sub

	sub_shape = np.array(arr[0]).shape

	# pad all sub arrays to longest sub array length with last subarray values
	matrix = np.empty((0, *(() if len(sub_shape) == 1 else np.array(arr[0]).shape[:-1]), l))

	for sub in arr:
		sub = np.array(sub)
		l_sub = sub.shape[-1]

		if l-l_sub is not 0:
			if len(sub_shape) == 1:
				padded_sub = np.append(sub, np.repeat(sub[..., -1], l-l_sub))
			else:
				padded_sub = np.hstack((sub, np.tile(sub[:, [-1]], l-l_sub)))
		else:
			padded_sub = sub
		matrix = np.append(matrix, np.array([padded_sub]), axis=0)

	return matrix


def perParticipantDict(X, Y, P):
	participants_dict = {}

	for i, _id_ in enumerate(P):
		if _id_ not in participants_dict:
			participants_dict[_id_] = ([], [])	# (P_X, P_Y)

		participants_dict[_id_][0].append(X[i])
		participants_dict[_id_][1].append(Y[i])

	for k in participants_dict:
		X, Y = participants_dict[k]
		participants_dict[k] = (np.array(X), np.array(Y))

	return participants_dict


def keras_base_model(model):
	model_cpy = tf.keras.models.clone_model(model)
	model_cpy.build(model.input.shape)
	model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def keras_model_cpy(model):
	model_cpy.set_weights(model.get_weights())

	return model_cpy


# curve: n x |max(C_tr)|
# sw_rw: ((sw_avg_f1, sw_std_f1),(rw_avg_f1, rw_std_f1))
def standard_F1_Ctr_graph(curve, sw_rw, title_label=None, sw_rw_labels=True):
	x_axis = list(range(curve.shape[-1]))

	curve = curve[~np.isnan(curve).any(axis=1)]

	avg_calib_f1 = np.nanmean(curve, axis=0)
	std_calib_f1 = np.nanstd(curve, axis=0)

	plt.plot(avg_calib_f1)

	plt.fill_between(
		x_axis,
		avg_calib_f1-std_calib_f1,
		avg_calib_f1+std_calib_f1,
		alpha=0.4
	)

	# graph end points with their error bars
	sw, rw = sw_rw
	sw_avg_f1, sw_std_f1 = sw
	rw_avg_f1, rw_std_f1 = rw

	plt.plot(0, sw_avg_f1, 'go', label='subject-wise' if sw_rw_labels else None)
	plt.errorbar(0, sw_avg_f1, ecolor='green', yerr=sw_std_f1, capsize=10)

	plt.plot(x_axis[-1], rw_avg_f1, 'ro', label='random-wise' if sw_rw_labels else None)
	plt.errorbar(x_axis[-1], rw_avg_f1, ecolor='red', yerr=rw_std_f1, capsize=5)

	plt.grid(linestyle='--', linewidth=0.5)

	if title_label is not None:
		plt.title(title_label)


def collapse_P(d):
	out = []

	for p_id in d.keys():
		if len(d[p_id].shape) == 2:		# calib_seed
			out.append(d[p_id])
		elif len(d[p_id].shape) == 3:	# calib_cv
			for curve in d[p_id]:
				out.append(curve)

	return pad_last_dim(out)
