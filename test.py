import unittest

# for tests
import numpy as np
from random import seed, randint
import tensorflow as tf
from tensorflow.keras.layers import Dense

from load_data import load_surface_data, _CACHED_load_surface_data

# exported methods
from util_functions import partic_calib_curve, all_partic_calib_curve
from util_functions import pcc_cv, all_pcc_cv

import util_functions as PaCalC

# ==========
# dataset for tests

global _cached_Irregular_Surface_Dataset
_cached_Irregular_Surface_Dataset=None

import time

s = time.time()
X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1, consent=False)
e = time.time()

print('TIME TO HIT CACHE & SERVE:'+str(e-s)+'s')

seed(39)
np.random.seed(39)

# ==========

class PaCalC_exported_func(unittest.TestCase):

	def test_partic_calib_curve(self):
		
		matrix = partic_calib_curve(TestHelperFunc.make_model(), *TestHelperFunc.P_XY())
		
		# self.assertEqual(matrix.shape, (2,6))
		self.assertEqual(matrix.shape, (2,8))

	def test_all_partic_calib_curve(self):		
		D = all_partic_calib_curve(TestHelperFunc.make_model(), *TestHelperFunc.XYP())
		
		self.assertEqual(len(D.keys()), 5)

		for matrix in D.values():
			# self.assertEqual(matrix.shape, (2,2))
			self.assertEqual(matrix.shape, (1,2,2))

	def test_cv_single_partic(self):
		cv = 2
		
		matrix = pcc_cv(TestHelperFunc.make_model(), *TestHelperFunc.P_XY(), cv=cv)
		
		# self.assertEqual(matrix.shape, (cv,2,6))
		self.assertEqual(matrix.shape, (cv,2,8))

	def test_cv_all_partic(self):
		cv = 2
		
		D = all_pcc_cv(TestHelperFunc.make_model(), *TestHelperFunc.XYP(), cv=cv)
		
		for matrix in D.values():
			# self.assertEqual(matrix.shape[:2], (cv, 2))
			self.assertEqual(matrix.shape[:2], (cv, 1))

class TDD_PaCalC(unittest.TestCase):

	def test_perLabelDict(self):
		P_X, P_Y = TestHelperFunc.P_XY()
		d,_ = PaCalC.perLabelDict(P_X, P_Y)

		for i, p_x in enumerate(d.values()):
			in_PX = np.array(P_X[i*25:(i*25+25),:])
			out_PX = np.array(p_x)

			self.assertTrue((out_PX == in_PX).all())

	def test_pad_last_dim(self):
		n_labels = 10
		f1_curves_per_label = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*i
			i += 1
			f1_curves_per_label.append(f1_curve)

		F1 = PaCalC.pad_last_dim(f1_curves_per_label)

		self.assertEqual(F1.shape, (n_labels,n_labels))

	def test_pad_last_dim_matrices(self):
		n_labels = 10
		f1_1 = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*i
			i += 1
			f1_1.append(f1_curve)

		f1_2 = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*(i+1)
			i += 1
			f1_2.append(f1_curve)

		f1_curves = [PaCalC.pad_last_dim(f1_1), PaCalC.pad_last_dim(f1_2)]

		F1 = PaCalC.pad_last_dim(f1_curves)

		self.assertEqual(F1.shape, (2,n_labels,n_labels+1))

	def test_perParticipantDict(self):
		X,Y,P = TestHelperFunc.XYP()
		d = PaCalC.perParticipantDict(X,Y,P)

		for i, xy in enumerate(d.values()):
			x, y = xy
			self.assertTrue((x == np.array(X[i::5,:])).all())
			self.assertTrue((y == np.array([[1,0]]*5+[[0,1]]*5)).all())

	# def test_P_missing_labels():
	# 	P_X, P_Y = np.random.rand(50,100), np.array([[1,0,0,0,0,0,0,0,0]]*25+[[0,1,0,0,0,0,0,0,0]]*25)

	# 	model = tf.keras.models.Sequential()
	# 	model.add(tf.keras.layers.Dense(32, input_dim=100, activation='relu'))
	# 	model.add(tf.keras.layers.Dense(16, activation='relu'))
	# 	model.add(tf.keras.layers.Dense(P_Y.shape[-1], activation='softmax'))
	# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# 	matrix = PaCalC.partic_calib_curve(model, P_X, P_Y)

	# 	matrix = np.array([matrix])

	# 	PaCalC.graph_calib_curve_general(matrix)

	# 	PaCalC.graph_calib_curve_per_Y(matrix)

	# def test_all_missing_labels():
	# 	X, Y, P = np.random.rand(50,100), np.array([[1,0,0,0,0,0,0,0,0]]*25+[[0,1,0,0,0,0,0,0,0]]*25), np.array([1,2,3,4,5]*10)

	# 	model = tf.keras.models.Sequential()
	# 	model.add(tf.keras.layers.Dense(32, input_dim=100, activation='relu'))
	# 	model.add(tf.keras.layers.Dense(16, activation='relu'))
	# 	model.add(tf.keras.layers.Dense(Y.shape[-1], activation='softmax'))
	# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# 	d = PaCalC.all_partic_calib_curve(model, X, Y, P)

	# 	matrix = PaCalC.collapse_P(d)

	# 	PaCalC.graph_calib_curve_general(matrix)

	# 	PaCalC.graph_calib_curve_per_Y(matrix)

	# 	for p_id, p_curves in d.items():
	# 		print(f'P id: {p_id}')
	# 		# p_curves = np.array([p_curves])
	# 		PaCalC.graph_calib_curve_general(np.array([p_curves]), p_id)
	# 		PaCalC.graph_calib_curve_per_Y(np.array([p_curves]), p_id)

class TestHelperFunc:
	def make_model():
		model = tf.keras.models.Sequential()
		model.add(Dense(32, input_dim=100, activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam',loss='mean_squared_error')
		return model

	def P_XY():
		P_X, P_Y = np.random.rand(50,100), np.array([[1,0]]*25+[[0,1]]*25) # replace None with 25 0's & 25 1's both ohe
		return P_X, P_Y

	def XYP():
		X, Y, P = np.random.rand(50,100), np.array([[1,0]]*25+[[0,1]]*25), np.array([1,2,3,4,5]*10)
		return X, Y, P

if __name__ == '__main__':
	unittest.main()