import numpy as np


def linear_modes(x, y):
    cov = x.T @ x

    # Add regularization
    alpha = 0.000001  # A small "regularization" parameter
    I = np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov + alpha * I)
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
    return M


def create_mask(modes, digit, alpha):
    w = modes @ digit.T
    mask = np.zeros((28.28))
    for i in range(10):
        m = modes[i].reshape(28, 28)
        mask += w[i]*m



def create_sorter(modes_list, img_size, phase_only=False):

    # Creating the modes
    max_x = 4
    max_y = 4
    x = np.linspace(-max_x, max_x, img_size)
    y = np.linspace(-max_y, max_y, img_size)
    X, Y = np.meshgrid(x, y)

    alpha = img_size / (2*len(modes_list)*max_x)
    beta = img_size / (2*len(modes_list)*max_y)

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