import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

x_train = np.load('../data/x_train.npy')
y_train = np.load('../data/y_train.npy')

x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

if x_train.ndim == 3:
    x_train = x_train.reshape(len(x_train), -1)
x_train = x_train.astype(np.float32)/255
if x_test.ndim == 3:
    x_test = x_test.reshape(len(x_test), -1)
x_test =x_test.astype(np.float32)/255



lda = LDA(n_components=9, solver='svd')
X_proj = lda.fit_transform(x_train, y_train)

# Projection matrix W: shape (n_features, C-1) → here (784, ≤9). You used n_components=2.
W = lda.scalings_[:, :9]   # take the first 2 discriminant vectors
print("W shape:", W.shape) # (784, 2)


# zero-mean, unit-norm each vector (helps visual compare)
Wn = W.copy()
Wn -= Wn.mean(axis=0, keepdims=True)
Wn /= (Wn.std(axis=0, keepdims=True) + 1e-12)

vlim = np.abs(Wn).max()

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(Wn[:, i].reshape(28, 28), cmap="bwr", vmin=-vlim, vmax=vlim)
    ax.set_title(f"LDA {i+1}")
    ax.axis('off')
fig.suptitle("LDA discriminant vectors (shared scale)")
plt.tight_layout()
plt.show()



# Optional: how much each axis separates the classes
print("explained_variance_ratio_:", lda.explained_variance_ratio_)

