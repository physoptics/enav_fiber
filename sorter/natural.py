import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sorter_utils.help_functions import *

############################ Data load##################################
# X = np.load('../data/x_train.npy')
# Y = np.load('../data/y_train.npy')
X = np.load('../../mnist/x_train.npy')
Y = np.load('../../mnist/y_train.npy')
X_flat = X.reshape(len(X), -1)
X_flat = X_flat.astype(np.float32)
scaler = StandardScaler(with_mean=True, with_std=True)
X_std = scaler.fit_transform(X_flat)
one_hot_Y = np.eye(10)[Y]

M = linear_modes(X_std, one_hot_Y)










