import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from enav_fiber.utils.boldoc import slm_bolduc_create_mask
from enav_fiber.utils.circ import create_circle
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
mask = create_mask(M, X_std[3], 1080)
print(Y[3])
g = gaussian(1080)

# mask = phase(1080, 1)
g = center_image(g,1920, 1080)
amp = np.abs(mask)
ang = np.angle(mask)

plt.imshow(g)


# m = create_circle(1920, 1080, R=150, shiftx=960, shifty=540)
# big = paste_image(mask, 1920, 1080, x_offset=960, y_offset=540, fill_value=0)
amp = center_image(amp,1920, 1080)*g
ang = center_image(ang,1920, 1080)

M = slm_bolduc_create_mask(ang, amp)
cv2.imshow("SLM",M)
cv2.moveWindow("SLM", 1920, 0)  # move to monitor 2 (adjust as needed)
cv2.waitKey(100000000)








#
plt.imshow(g)
plt.show()
# print(Y[9])
#
# Iout = np.abs(np.fft.fftshift(np.fft.fft2(mask))**2)
# plt.imshow(Iout)
# plt.show()
#
# diag = np.diag(Iout)
# plt.plot(diag)
# plt.show()
#
# # plt.figure()
# # Use cmap='gray' for grayscale images like MNIST
#
# # plt.imshow(np.real(mask).astype(float), cmap='gray')
# # plt.imshow(mm, cmap='gray')
# # plt.title("2D Array Plot (Image)")
# # plt.colorbar()  # Adds a color bar to show the value scale
# # plt.show()









