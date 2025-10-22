import numpy as np
import cv2
import matplotlib.pyplot as plt

x_train = np.load(r"C:\Users\enav\PycharmProjects\fiber_slm\data\x_train.npy")



def duplicate( digit, we, rand):
    img84 = cv2.resize(digit,(84, 84), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((336, 336), dtype=digit.dtype)
    ww = np.zeros((336,336))
    for i in range(8):
        for j in range(8):
            ww[i*42:42*(i+1),j*42:42*(j+1)] = we[42*i, 42*j] *rand[8*i+j]
    for i in range(3):
        for j in range(3):
            out[42+84*i:42+84*(i+1), 42+84*j:42+84*(j+1)] = img84

    out[0:42, 126: 210] = img84[0:42, :]
    out[294:, 126: 210] = img84[42:, :]
    out[126: 210, 0:42] = img84[:, 0:42]
    out[126: 210, 294:] = img84[:, 42:]
    return out +ww

# out = duplicate(x_train[9],w, random)
# plt.figure()
# plt.imshow(out)
# plt.axis('off')
# plt.tight_layout()
# plt.show()