import numpy as np
import cv2

x_train = np.load('../enav_fiber/data/x_train.npy')

def duplicate( digit, w):
    img84 = cv2.resize(digit,(84, 84), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((336, 336), dtype=digit.dtype)
    for i in range(3):
        for j in range(3):
            out[42+84*i:42+84*(i+1), 42+84*j:42+84*(j+1)] = img84
    out[0:42, 84: 168] = img84[0:42, :]
    out[294:, 84: 168] = img84[42:, :]
    return out


