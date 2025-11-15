import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def linear_modes(x, y):
    cov = x.T @ x
    print(cov.shape)
    # Add regularization
    alpha = 1e-8  # A small "regularization" parameter
    I = np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov + alpha * I)
    inv_cov = inv_cov / np.max(inv_cov)
    M = (y.T @ x) @ (inv_cov)
    es_y = (M @ x.T).T
    max_indices = np.argmax(es_y, axis=1)
    es_y = np.zeros(es_y.shape)
    es_y[np.arange(len(es_y)), max_indices] = 1.0
    error_y = y - es_y
    error_y = np.abs(error_y)
    error_y = np.sum(error_y, axis=1)
    error_y = np.sum(error_y, axis=0)
    error_y = error_y / 120000

    print(1 - error_y)
    return M/np.max(M)

def PPCA(x,y,k):
    n_features = x.shape[1]
    pca = PCA(n_components=n_features, svd_solver='full')
    X_pca = pca.fit_transform(x)  # scores: (n, n_components)
    p = pca.components_[:k]  # loadings: (n_components, 784)
    explained = pca.explained_variance_ratio_  # (n_components,)
    VT = p @ x.T
    V = VT.T
    VTV = VT @ V
    inv_VTV = np.linalg.inv(VTV)
    M = (y.T @ V) @ (inv_VTV)
    return M, p

def phase(N, a):
    # 1. Create 1D coordinate vectors
    # We'll create a normalized grid from 0 to 1
    t = np.linspace(0, 1, N)

    # 2. Create 2D coordinate grids (X and Y)
    # Both X and Y will be (28, 28) arrays
    X, Y = np.meshgrid(t, t)
    phase_linear = 2 * np.pi * (a * X + a * Y)
    # Calculate the imaginary exponent
    complex_array_linear = np.exp(1j * phase_linear)
    return complex_array_linear

def create_mask(modes, digit, N):
    new_size = (N, N)
    w = modes @ digit.T
    mask = np.zeros(new_size, dtype=complex)
    for i in range(10):
        m = modes[i].reshape(28, 28)
        alpha = int((N/10)*i)
        exp = phase(N,alpha)
        # mm = cv2.resize(m, new_size, interpolation=cv2.INTER_LINEAR)
        mask += (w[i]*exp)
    mask = mask / np.max(np.abs(mask))
    return mask


def gaussian(N=1080):
    # 1. Create the coordinate grid
    t = np.linspace(0, N - 1, N)  # Coordinates go from 0 to N-1
    X, Y = np.meshgrid(t, t)

    # 2. Determine the center of the grid

    center = (N - 1) / 2
    X = X - center
    Y = Y - center
    Xprime=X*np.cos(-0.21)+Y*np.sin(-0.21)
    Yprime=-X*np.sin(-0.21)+Y*np.cos(-0.21)

    # 3. Calculate the squared distance from the center, applying scaling factors
    # The (X - center) and (Y - center) terms shift the peak to the middle.
    gauss = ((Xprime ) / 712.5) ** 2 + ((Yprime ) / 612.5) ** 2
    gauss2 = ((Xprime) / 100.0) ** 2 + ((Yprime) / 100.0) ** 2

    # 4. Calculate the Gaussian
    # The standard Gaussian is exp(-gauss), which is centered and peaks at 1.
    arr = np.exp(-gauss)
    arr2 = np.exp(-gauss2)

    # 5. Your function includes an inversion (1. / arr).
    # This turns the peak into a valley (an inverted Gaussian).
    arr = 1 / arr
    arr = arr/ np.max(arr)
    arr = arr*arr2


    return arr2




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


def center_image(img: np.ndarray,
                big_width: int,
                big_height: int,
                fill_value: int = 0) -> np.ndarray:
    h, w = img.shape
    x_offset = (big_width - w)//2
    y_offset = (big_height - h)//2
    arr = np.full((big_height, big_width),fill_value, dtype=img.dtype)
    arr[y_offset:y_offset+h, x_offset:x_offset+w] = img[:, :]
    return arr
