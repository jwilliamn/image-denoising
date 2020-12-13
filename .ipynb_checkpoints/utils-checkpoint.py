import numpy as np
import matplotlib.pyplot as plt

def calc_unobserved_rmse(A, A_hat, mask):
    """
    Calculate RMSE on all unobserved entries in mask, for true matrix UVᵀ.
    Parameters
    ----------
    A : m x n array
        true  matrix
    A_hat : m x n array
        estimated matrix
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    Returns:
    --------
    rmse : float
        root mean squared error over all unobserved entries
    """
    pred = np.multiply(A_hat, (1 - mask))
    truth = np.multiply(A, (1 - mask))
    cnt = np.sum(1 - mask)
    return (np.linalg.norm(pred - truth, "fro") ** 2 / cnt) ** 0.5


def plot_image(A, title=None):
    plt.imshow(A, cmap='gray')
    plt.title(title, fontsize=15)
    
def plot_proc(img1, img2, title):
    # Plot original image and denoised
    plt.figure(figsize=[20, 10])
    plt.subplot(121)
    plot_image(img1, 'True Image')

    plt.subplot(122)
    plot_image(img2, title)