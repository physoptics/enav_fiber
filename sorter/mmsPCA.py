import numpy as np
from sklearn.datasets import fetch_openml, load_digits
import matplotlib.pyplot as plt

def create_sorter(modes_list, img_size, phase_only=False):

    # Creating the modes
    max_x = 4
    max_y = 4
    x = np.linspace(-max_x, max_x, img_size)
    y = np.linspace(-max_y, max_y, img_size)
    X, Y = np.meshgrid(x, y)

    alpha = img_size / (2*len(modes_list)*max_x)
    beta = img_size / (2*len(modes_list)*max_y)


    # Plotting the modes
    fig, axs = plt.subplots(1, len(modes_list), figsize=(15, 3))
    for i, mode in enumerate(modes_list):
        ax = axs[i]
        ax.imshow(np.abs(mode), extent=[x.min(), x.max(), y.min(), y.max()], cmap='jet')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Creating the SLM
    M = np.zeros((img_size, img_size), dtype=complex)
    for i, mode in enumerate(modes_list):
        loc = ( alpha * (i + 0.5 - len(modes_list) / 2) \
              , beta * (i + 0.5 - len(modes_list) / 2) )
        M += np.conjugate(mode) * np.exp(1j * np.pi * (loc[0]*X+ loc[1]*Y))
    M /= np.max(np.abs(M))

    # Phase only
    if phase_only:
        real_M = np.real(M)
        imag_M = np.imag(M)
        phi = np.arctan2(imag_M, real_M)
        M = np.exp(1j * phi)


    # Show the mask
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(M), extent=[x.min(), x.max(), y.min(), y.max()], cmap='jet')
    plt.title('SLM Amplitude Mask')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(M), extent=[x.min(), x.max(), y.min(), y.max()], cmap='jet')
    plt.title('SLM Phase Mask')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Display the projection
    Iout_per_mode = []
    plt.figure(figsize=(15, 3))
    for i, mode in enumerate(modes_list):
        plt.subplot(1, len(modes_list), i + 1)
        Iout = np.abs(np.fft.fftshift(np.fft.fft2(mode * M)))**2
        plt.imshow(Iout, extent=[-max_x, max_x, -max_y, max_y], cmap='hot')
        # plt.title(f'Projection {modes_list[i]}')
        # Add text
        plt.text(0.0, -0.8*max_y, f'{i}', fontsize=12, ha='center', va='center', color='white')
        plt.axis('off')
        Iout_per_mode.append(Iout)
    plt.tight_layout()
    plt.show()

    return M, Iout_per_mode


import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- Parameters ---
M = 10           # number of components
n_train = 10_000
method = 'PCA'   # choose: 'PCA' or 'LDA'
dataset_name = 'FMNIST'

# --- Load DATASET ---
print(f"Loading {dataset_name} dataset...")
if dataset_name == 'MNIST':
    data = fetch_openml('mnist_784', version=1, as_frame=False)
    X = data['data'].astype(np.float64) / 255.0
    y = data['target'].astype(int)
elif dataset_name == 'Digits':
    data = load_digits(n_class=M)
    X, y = data.data, data.target
elif dataset_name == 'FMNIST':
    data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)

# --- Normalize the data ---

X = X / 255.0
X -= X.mean(axis=0)
X /= np.sqrt(np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

X = X[:n_train]
y = y[:n_train]
num_classes = len(np.unique(y))
M = min(M, num_classes)  # LDA can't have more than (#classes - 1) components

# --- One-hot encoding ---
Y_onehot = np.zeros((y.shape[0], num_classes))
for i, label in enumerate(y):
    Y_onehot[i, label] = 1.0

# --- Dimensionality Reduction ---
# --- Dimensionality Reduction ---
print(f"Applying {method}...")

if method == 'PCA':
    # Step 1: Compute covariance matrix
    X_centered = X - np.mean(X, axis=0)
    C = (X_centered.T @ X_centered) / X_centered.shape[0]

    # Step 2: Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:M]]  # top M eigenvectors

    # Step 3: Project data onto PCA basis
    P = X_centered @ eigvecs

elif method == 'LDA':
    # Adjust M automatically to avoid error
    M = min(M, num_classes - 1)
    print(f"LDA components limited to {M} (since there are {num_classes} classes).")
    lda = LinearDiscriminantAnalysis(n_components=M)
    P = lda.fit_transform(X, y)
    eigvecs = lda.scalings_[:, :M] if lda.scalings_ is not None else None

else:
    raise ValueError("method must be 'PCA' or 'LDA'")


# --- Step 4: Solve least squares for classifier W ---
print("Solving least squares classifier...")
Y_padded = np.zeros((Y_onehot.shape[0], M))
Y_padded[:, :min(num_classes, M)] = Y_onehot[:, :min(num_classes, M)]

PTP_inv = np.linalg.inv(P.T @ P + 1e-8 * np.eye(P.shape[1]))
W = PTP_inv @ (P.T @ Y_padded)  # (M x M or fewer)

print(f"{method} completed. Shape of features: {P.shape}, W: {W.shape}")


# --- Step 5: Closest unitary approximation ---
# # SVD: W = U Σ V^T, closest unitary = U V^T
# U_svd, _, Vt = np.linalg.svd(W)
# U_unitary = U_svd @ Vt
U_unitary = W   # TODO

# --- Step 6: Prediction function ---
def predict(X_data, eigvecs, U_unitary):
    # Xc = X_data - np.mean(X, axis=0)
    P_data = X_data @ eigvecs
    out = P_data @ U_unitary

    # # Not normalize because the mask is amplitude + phase!
    # print("Output norms (first 6):", np.linalg.norm(out[:6], axis=1))

    out = np.abs(out)**2  # Measure the intensity
    preds = np.argmax(out[:, :num_classes], axis=1)
    return preds, out

# --- Train/Test Split (manual) ---
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Predict
y_pred, out = predict(X_test, eigvecs, U_unitary)

# Plot bars of the classification for some examples
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    # The correct bar should be red
    if method == 'LDA':
        bars = plt.bar(range(M), out[i, :M])
        bars[np.argmax(out[i, :M])].set_color('g')
        # bars[y_test[i]].set_color('r')
    else:
        bars = plt.bar(range(num_classes), out[i, :num_classes])
        # The maximum bar should be green
        bars[np.argmax(out[i, :num_classes])].set_color('g')
        # The rest should be blue
        bars[y_test[i]].set_color('r')
    plt.title(f"True: {y_test[i]}, Pred: {y_pred[i]}")
plt.xlabel("Class")
plt.ylabel("Output intensity")
plt.tight_layout()
plt.show()
# Show the corresponding images
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    # plt.imshow( (np.abs(X_test[i])+ mean_shift).reshape(8, 8), cmap='gray')
    data.data[i] -= data.data[i].min()
    data.data[i] = data.data[i] / data.data[i].max()
    s = int(data.data[i].shape[0]**0.5)
    plt.imshow( (data.data[i]).reshape(s, s), cmap='gray')
    plt.axis('off')
plt.title(f"Label: {y_test[i]}")
plt.tight_layout()
plt.show()


acc = np.mean(y_pred == y_test)
print(f"Test accuracy with unitary classifier (M={M}): {acc:.4f}")

# Get the Training accuracy:
y_pred, out = predict(X_train, eigvecs, U_unitary)
acc = np.mean(y_pred == y_train)
print(f"Train accuracy with unitary classifier (M={M}): {acc:.4f}")


# The modes are the
modes = eigvecs @ U_unitary
modes_list = []
img_size = s*5
for i in range(modes.shape[1]):
    mode = modes[:, i].reshape(s, s)   # for FMNIST
    # Make upscaling to img_width x img_width
    mode = np.kron(mode, np.ones((img_size//s, img_size//s)))
    mode /= np.sqrt( (np.abs(mode)**2).sum() )  # normalize
    E = np.abs(mode)
    phase = np.angle(mode)
    modes_list.append( E * np.exp(1j * phase) )

# Create the sorter mask
M_mask, Iout_per_mode = create_sorter(modes_list, img_size=img_size, phase_only=False)