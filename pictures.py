# view_out_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt

# Load without copying to RAM

# arr = np.load(r"D:\fiber_data\enav\fiber_array.npy", mmap_mode="r")
data = np.load(r"D:\fiber_data\enav\fiber_pair.npz")
arr1, arr = data["arr1"], data["arr2"]
# print("loaded:", arr.shape, arr.dtype)

plt.figure()
plt.imshow(arr[5])
plt.axis('off')
plt.tight_layout()
plt.show()

