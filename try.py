import os,numpy as np
import cv2
import time
from utils.circ  import *
from utils.boldoc  import *
from utils.display  import *
from utils.camera_t import *
import matplotlib.pyplot as plt
x_train = np.load('../data/x_train.npy')
y_train = np.load('../data/y_train.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

cam = camera_setup()

t0 = time.perf_counter()




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

x_train_resized = np.zeros((N, 450, 450), dtype=x_train.dtype)
out = np.zeros((N, 144, 144), dtype=np.uint32)
for i in range(N):
    # cv2.resize expects H×W in pixels, and returns same dtype
    x_train_resized[i] = cv2.resize(
        x_train[i],
        (450, 450),
        interpolation=cv2.INTER_CUBIC
    )
for i in range(N):
    big = paste_image(x_train_resized[i], 1920, 1080, x_offset=1215, y_offset=315, fill_value=0)
    M = slm_bolduc_create_mask(big, mask)
    cv2.imshow("SLM", M)
    cv2.moveWindow("SLM", 1920, 0)  # move to monitor 2 (adjust as needed)
    if cv2.waitKey(150) & 0xFF == 27:  # 3000 ms
        break
    # out[i] = camera_capture(cam)
    p = camera_capture(cam)
    plt.figure()

    plt.imshow(p)
    plt.axis('off')
    plt.show()




os.makedirs("data", exist_ok=True)
np.save("data/out_uint32.npy", out)   # writes ~5.24 GB for your shape/dtype

# later (without loading all into RAM):
out_mmap = np.load("data/out_uint32.npy", mmap_mode="r")

cam.Exit()
cv2.destroyAllWindows()
t_total = time.perf_counter() - t0
print (t_total)













