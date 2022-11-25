# script for py_compatible_data.mat => X.npy
import scipy.io
import numpy as np

def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot, unique

data = scipy.io.loadmat('../py_compatible_data.mat')
gait_ireg_surface_dataset = data['gait_ireg_surface_dataset']

one_hot_labels, unique = one_hot([e[2][0][0][0] for e in gait_ireg_surface_dataset])
print('Labels generated')
participantID = np.array([g[0][0][0] for g in gait_ireg_surface_dataset])
print('Participant ID generated')
features = np.array([g[1] for g in gait_ireg_surface_dataset])
print('Features generated')

print(unique)

np.save('../labels', unique)
np.save('../GoIS_P', participantID)
np.save('../GoIS_X', features)
np.save('../GoIS_Y', one_hot_labels)

# code to load in np array
# gait_ireg_surface_dataset = np.load('gait_ireg_surface_dataset.npy', allow_pickle=True)

