import os,numpy as np
import cv2
import time
from utils.circ  import *
from utils.boldoc  import *
from utils.dup  import *
from utils.display  import *
from utils.camera_t import *
import matplotlib.pyplot as plt
x_train = np.load('../data/x_train.npy')
y_train = np.load('../data/y_train.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

cam = camera_setup()

t0 = time.perf_counter()


w = weight = np.ones((336, 336), dtype=x_train.dtype)
w[0:42,210:]= w[0:42,:126]=w[294:,210:]=w[294:,:126]= 0
w[210:,0:42] = w[:126,0:42] = w[210:,294:] = w[:126,294:]=0



def paste_image(img: np.ndarray,
                big_width: int,
                big_height: int,
                x_offset: int,
                y_offset: int,
                fill_value: int = 0) -> np.ndarray:

    # Determine channels
    if img.ndim == 3:
        channels = img.shape[2]
        canvas = np.full((big_height, big_width, channels),
                         fill_value, dtype=img.dtype)
    else:
        canvas = np.full((big_height, big_width),
                         fill_value, dtype=img.dtype)

    h, w = img.shape[:2]
    # Paste
    canvas[y_offset:y_offset+h, x_offset:x_offset+w, ...] = img
    return canvas


mask = create_circle(1920, 1080, R=300, shiftx=-480, shifty=-20)




# suppose x_train is shape (N, 28, 28), dtype float32 in [0,1]
N = 1000
for k in range(6):
    for j in range(60):
        x_save = np.zeros((N, 28, 28), dtype=x_train.dtype)
        x_train_resized = np.zeros((N, 336, 336), dtype=x_train.dtype)
        rand = np.zeros((N,144))
        out = np.zeros((N, 144, 144), dtype=np.uint32)
        for i in range(N):
            rand[i] = np.random.rand(144)
            # cv2.resize expects H×W in pixels, and returns same dtype
            x_save[i] = x_train[1000*j+ i]
            x_train_resized[i] =duplicate( x_train[1000*j+ i], w, rand[i])
        for i in range(N):
            big = paste_image(x_train_resized[i], 1920, 1080, x_offset=1272, y_offset=372, fill_value=0)
            M = slm_bolduc_create_mask(big, mask)
            cv2.imshow("SLM", M)
            cv2.moveWindow("SLM", 1920, 0)  # move to monitor 2 (adjust as needed)
            if cv2.waitKey(150) & 0xFF == 27:
                break
            out[i] = camera_capture(cam)


        save_path = r"D:\fiber_data\enav"

        # os.makedirs(save_path, exist_ok=True)
        # np.save(os.path.join(save_path, "fiber_array.npy"),out)

        filename = os.path.join(save_path, f"fiber_pair_label_{(k+3):03d}_{j:03d}.npz")
        np.savez_compressed(filename, weights= rand, x = x_save,  output =out)


cam.Exit()
cv2.destroyAllWindows()
t_total = time.perf_counter() - t0
print (t_total)












