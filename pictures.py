# view_out_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt

# Load without copying to RAM
arr_int32 = np.load("data/out_uint32.npy", mmap_mode="r")  # shape (N, 1024, 1280)
# print("loaded:", arr.shape, arr.dtype)

print(arr_int32[0].dtype, arr_int32[0].shape, arr_int32[0].min(), arr_int32[0].max())
u = np.unique(arr_int32)
print("unique (sample):", u[:10], "… total:", u.size)

plt.figure()
plt.imshow(arr[0], cmap='gray', vmin=arr[0].min(), vmax=arr[0].max())
plt.axis('off')
plt.tight_layout()
plt.show()

