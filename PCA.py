import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================
# 1) Load/prepare your MNIST
# ==============================
# Replace these with your actual np arrays
# X: (n, 28, 28) or (n, 784), y: (n,)
X = np.load('../data/x_train.npy')
Y = np.load('../data/y_train.npy')


assert X.ndim in (2, 3), "X must be (n,784) or (n,28,28)"
if X.ndim == 3:
    H, W = X.shape[1], X.shape[2]
    assert H == 28 and W == 28, "Expected 28x28 MNIST"
    X_flat = X.reshape(len(X), -1)
else:
    X_flat = X
    assert X_flat.shape[1] == 28*28, "Expected 784 features for MNIST"

# Optional: convert to float and normalize scale
X_flat = X_flat.astype(np.float32)

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

print(f"Total explained variance: {explained.sum():.4f}")

# ==============================
# 4) Scree + cumulative EV
# ==============================

cum_exp = np.cumsum(explained)
plt.figure(figsize=(6,4))
plt.plot(explained + 1e-12, lw=1)   # per-component EVR (add tiny eps to avoid log(0))
plt.plot(cum_exp, lw=1)             # cumulative EVR ~ near 1 (usually keep linear)
plt.yscale('log')                   # <-- log scale on y
plt.title("Explained Variance (log y)")
plt.xlabel("Component index")
plt.ylabel("Variance ratio (log)")
plt.legend(["Per-component", "Cumulative"])
plt.tight_layout()

# plt.plot(explained, lw=1)
# plt.plot(cum_exp, lw=1)
# plt.title("PCA Explained Variance (per PC) and Cumulative")
# plt.xlabel("Component index")
# plt.ylabel("Variance ratio")
# plt.legend(["Per-component", "Cumulative"])
# plt.tight_layout()

# =========================================
# 5) Visualize the first K eigen-digits
#    (principal component vectors as images)
# =========================================
def show_components_grid(components, K=50, rows=5, cols=10):
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

show_components_grid(components, K=50, rows=5, cols=10)

# =========================================
# 6) Scatter in PC1–PC2 (color by label)
#    Downsample if you have many points
# =========================================
def scatter_pc12(X_pca, y, max_points=10000):
    n = len(X_pca)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)
    Z = X_pca[idx, :2]
    labels = y[idx]
    plt.figure(figsize=(6,6))
    # Create a simple palette: digits 0..9
    # (If you prefer default colors, you can skip setting c explicitly)
    plt.scatter(Z[:,0], Z[:,1], s=6, c=labels, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("MNIST in PCA space (PC1 vs PC2)")
    plt.tight_layout()

scatter_pc12(X_pca, Y, max_points=10000)

# =========================================
# 7) (Optional) 3D scatter PC1–PC3
# =========================================
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    def scatter_pc123(X_pca, y, max_points=6000):
        n = len(X_pca)
        if n > max_points:
            idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        else:
            idx = np.arange(n)
        Z = X_pca[idx, :3]
        labels = y[idx]
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=6, c=labels, alpha=0.75)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("MNIST in PCA space (3D)")
        plt.tight_layout()
    scatter_pc123(X_pca, y)
except Exception as e:
    print("3D scatter skipped:", e)

plt.show()
