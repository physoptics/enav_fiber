import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================
# 1) Load/prepare your MNIST
# ==============================
# Replace these with your actual np arrays
# X: (n, 28, 28) or (n, 784), y: (n,)
# X = np.load('../data/x_train.npy')
# Y = np.load('../data/y_train.npy')
X = np.load('../mnist/x_train.npy')
Y = np.load('../mnist/y_train.npy')

X_flat = X.reshape(len(X), -1)
X_flat = X_flat.astype(np.float32)

one_hot_Y = np.eye(10)[Y]
print(one_hot_Y.shape)
# ==============================
# 2) Standardize features
#    (mean 0 / std 1 per pixel)
# ==============================
scaler = StandardScaler(with_mean=True, with_std=True)
X_std = scaler.fit_transform(X_flat)

# ==============================
# 3) PCA (keep all components)
#    For MNIST: n_features=784
# ==============================
n_features = X_std.shape[1]
pca = PCA(n_components=n_features, svd_solver='randomized', random_state=0)
X_pca = pca.fit_transform(X_std)   # scores: (n, n_components)
components = pca.components_       # loadings: (n_components, 784)
explained = pca.explained_variance_ratio_  # (n_components,)

P = pca.components_[:200]

VT = P @ X_std.T
V = VT.T
VTV = VT @ V
inv_VTV = np.linalg.inv(VTV)
M = (one_hot_Y.T @ V)@ (inv_VTV)

es_y = (M @ VT).T
max_indices = np.argmax(es_y, axis=1)
print(es_y[10])
es_y = np.zeros(es_y.shape)
es_y[np.arange(len(es_y)), max_indices] = 1.0
print(es_y[10])
error_y = one_hot_Y - es_y
error_y = np.abs(error_y)
error_y = np.sum(error_y, axis=1)
error_y = np.sum(error_y, axis=0)
error_y = error_y/120000
print(error_y)












# =========================================
# 5) Visualize the first K eigen-digits

def show_components_grid(components, K=50, rows=5, cols=10):
    plt.figure(figsize=(6, 4))
    plt.plot(explained + 1e-12, lw=1)  # per-component EVR (add tiny eps to avoid log(0))
    plt.yscale('log')  # <-- log scale on y
    plt.title("Explained Variance (log y)")
    plt.xlabel("Component index")
    plt.ylabel("Variance ratio (log)")

    plt.tight_layout()
    K = min(K, components.shape[0])
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
    for i in range(rows*cols):
        ax = axes.flat[i]
        if i < K:
            img = components[i].reshape(28,28)
            # Normalize for better contrast in display
            vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"PC {i+1}")
        ax.axis('off')
    fig.suptitle(f"Top {K} principal components (eigen-digits)", y=0.92)
    plt.tight_layout()

# show_components_grid(components, K=50, rows=5, cols=10)




plt.show()
