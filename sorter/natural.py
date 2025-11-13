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

############################mode creator##################################
M = linear_modes(X_std, one_hot_Y)
# M2, p = PPCA(X_std, one_hot_Y,300)
# mm = M2[9].reshape(28,28)




############################mask creator##################################
mask = create_mask(M, X_std[99], 450)

plt.imshow(mask.real)
plt.show()
print(Y[9])

Iout = np.abs(np.fft.fftshift(np.fft.fft2(mask))**2)
plt.imshow(Iout)
plt.show()

diag = np.diag(Iout)
plt.plot(diag)
plt.show()

# plt.figure()
# Use cmap='gray' for grayscale images like MNIST

# plt.imshow(np.real(mask).astype(float), cmap='gray')
# plt.imshow(mm, cmap='gray')
# plt.title("2D Array Plot (Image)")
# plt.colorbar()  # Adds a color bar to show the value scale
# plt.show()









