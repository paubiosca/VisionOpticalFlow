import numpy as np
import cv2
import os

from flowToColorLib import flow_to_color

def compute_gradients(im1, im2):
    """
    Computes image gradients using the Horn-Schunck method.
    """
    Ix = 0.25 * (np.roll(im1, -1, axis=1) - np.roll(im1, 1, axis=1) + 
                 np.roll(im2, -1, axis=1) - np.roll(im2, 1, axis=1))
    Iy = 0.25 * (np.roll(im1, -1, axis=0) - np.roll(im1, 1, axis=0) + 
                 np.roll(im2, -1, axis=0) - np.roll(im2, 1, axis=0))
    It = 0.25 * (np.roll(im2, -1, axis=1) - np.roll(im1, -1, axis=1) + 
                 np.roll(im2, -1, axis=0) - np.roll(im1, -1, axis=0) +
                 np.roll(im2, -1, axis=(0,1)) - np.roll(im1, -1, axis=(0,1)))
    return Ix, Iy, It

def lucas(im1, im2, window_size):
    """
    Lucas-Kanade method for optical flow estimation.
    
    Parameters:
    im1: First image or frame at time t
    im2: Second image or frame at time t + 1
    window_size: Size of the window to consider around each pixel
    
    Returns:
    u: Optical flow horizontal component
    v: Optical flow vertical component
    """
    Ix, Iy, It = compute_gradients(im1, im2)

    # Initialize u and v optical flow vectors
    u = np.zeros(im1.shape)
    v = np.zeros(im2.shape)
    
    # Precompute the Gaussian window
    W = cv2.getGaussianKernel(window_size, -1) * cv2.getGaussianKernel(window_size, -1).T

    # Half window size for indexing
    half_w = window_size // 2

    # Iterate over all pixels except the border
    for i in range(half_w, im1.shape[0] - half_w):
        for j in range(half_w, im1.shape[1] - half_w):
            # Calculate windowed matrices
            Ix_w = Ix[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()
            Iy_w = Iy[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()
            It_w = -It[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1].flatten()

            # Apply the window weighting
            Ix_w = Ix_w * W.flatten()
            Iy_w = Iy_w * W.flatten()
            It_w = It_w * W.flatten()

            # Create matrix A and vector B
            A = np.vstack((Ix_w, Iy_w)).T
            B = It_w[:, np.newaxis]

            # Compute the least squares solution if A is not singular
            if np.linalg.matrix_rank(A) == 2:
                nu = np.linalg.inv(A.T @ A) @ A.T @ B
                u[i, j] = nu[0]
                v[i, j] = nu[1]
    return u, v

def read_image_placeholder(filename):
    # Read the image
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Convert to float
    im = np.float32(im) / 255.0
    
    return im

def main():
    # Get the current working directory
    cwd = os.getcwd()
    SCRIPTS_DIR = os.path.dirname(cwd)
    DATA_DIR = os.path.join(SCRIPTS_DIR, 'data', 'nasa')
    print('Data directory: ', DATA_DIR)
    
    I1 = read_image_placeholder(os.path.join(DATA_DIR, 'nasa9.png'))
    I2 = read_image_placeholder(os.path.join(DATA_DIR, 'nasa10.png'))

    # Set parameters for the algorithm
    window_size = 5  # the regularization parameter

    # Apply the Lucas-Kanade method
    u, v = lucas(I1, I2, window_size)
    
    # Use FlowToColor to visualize the optical flow
    flow_color_image = flow_to_color(np.dstack((u, v)))
    print('Saving output to data/lucas_output.png')
    cv2.imwrite(os.path.join(DATA_DIR, 'lucas_output.png'), flow_color_image)
    
    
if __name__ == '__main__':
    main()
