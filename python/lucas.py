import argparse
from scipy import signal
import numpy as np
import cv2
import os

from middlebury import computeColor
from utils import angular_error, end_point_error, get_GT_optical_flow_file, get_image_names, print_statistics, relative_norm_error, save_statistics_to_pickle

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

def lucas(im1, im2, window_size, kernel=None):
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
    velocity = np.zeros((Ix.shape[0], Ix.shape[1], 2)) # estimated velocity
    
    # Half window size for indexing
    half_w = window_size // 2

    # Iterate over all pixels except the border
    for i in range(half_w, im1.shape[0] - half_w):
        for j in range(half_w, im1.shape[1] - half_w):
            # Calculate windowed matrices
            Ix_w = Ix[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]
            Iy_w = Iy[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]
            It_w = It[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]

            if kernel is not None:
                 # Apply the window weighting   
                Ix_w = Ix_w*kernel
                Iy_w = Iy_w*kernel
                It_w = It_w*kernel

            # Create matrix A and vector B
            A = np.vstack((Ix_w.flatten(), Iy_w.flatten())).T
            B = (-It_w).flatten()

            # Compute the least squares solution if A is not singular
            if np.linalg.matrix_rank(A) == 2:
                nu = np.linalg.inv(A.T @ A) @ A.T @ B
                u[i, j] = nu[0]
                v[i, j] = nu[1]
                velocity[i, j] = [u[i, j], v[i, j]]
    return velocity

def read_image_placeholder(filename):
    # Read the image
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Convert to float  
    im = np.float32(im) / 255.0
    
    return im

def compute_gaussian_kernel(window_size, sigma):
    # Generate 1D Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(window_size, sigma)
    
    # Create 2D Gaussian kernel by taking the outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Normalize the kernel
    kernel_2d /= np.sum(kernel_2d)
    
    return kernel_2d

def main(args):

    # get params
    if __name__ == 'lucas':
        save_name_colormap = args["save_name_colormap"]
        dir = args["dir"]
        gaussian_kernel = args["gaussian_kernel"]
        window_min = args["window_min"]  # lower bound of the regularization parameter
        window_max = args["window_max"]
        window_step = args["window_step"]
        return_flow = args["return_flow"]
    else:
        save_name_colormap = args.save_name_colormap
        dir = args.dir
        window_min = args.window_min  
        window_max = args.window_max
        window_step = args.window_step
        return_flow = args.return_flow
        gaussian_kernel= args.gaussian_kernel  

    # Get the current working directory
    cwd = os.getcwd()
    SCRIPTS_DIR = os.path.dirname(cwd)
    DATA_DIR = os.path.join(SCRIPTS_DIR, 'data',dir)
    print('Data directory: ', DATA_DIR)
    
    image_name_1, image_name_2 = get_image_names(dir)
    print('Image names: ', image_name_1, image_name_2)
    I1 = read_image_placeholder(os.path.join(DATA_DIR, image_name_1))
    I2 = read_image_placeholder(os.path.join(DATA_DIR, image_name_2))

    statistics = {}
    flow = None
    for window in np.arange(window_min, window_max, window_step):
        print(f"\rwindow : {round(window, 5)}/{round(window_max, 5)}", end='')

        #Initialize window key
        statistics[window] = {}
        kernel = None
        if gaussian_kernel==True:
            # Precompute the Gaussian window
            kernel = compute_gaussian_kernel(window, 1.0)

        # Apply the Lucas-Kanade method
        flow = lucas(I1, I2, window, kernel) # estimated flow
        
        # Use FlowToColor to visualize the optical flow
        flow_color_image = computeColor(flow)
        print('Saving output to '+DATA_DIR+'/color_map_lucas_'+str(window)+'.png')
        cv2.imwrite(os.path.join(DATA_DIR+'/color_map_lucas_'+str(window)+'.png'), flow_color_image)
        
        # 7. If available: read the ground truth with the function readFlowFile() and compare with (uN , vN ) (see next subsection).
        gt_flow = get_GT_optical_flow_file(DATA_DIR)

        # 3. if a ground truth is available, you would compute several statistics (mean, standard deviation) of End-Point error, angular error, norm error. 
        if (gt_flow is not None):
            # Compute several errors
            error_functions = [ angular_error, end_point_error, relative_norm_error]
            errors= {}

            for error_func in error_functions:
                error_img = error_func(flow, gt_flow)
                mean_error = np.mean(error_img)
                std_error = np.std(error_img)

                # Initialize the dictionary if it doesn't exist
                errors[error_func.__name__] = {"mean": [], "std": []}
                
                errors[error_func.__name__]["mean"].append(mean_error)
                errors[error_func.__name__]["std"].append(std_error)
            
            # add to window key
            statistics[window] = errors.copy()
            
    # Display the statistics
    print_statistics(statistics)

    # save statistics into csv file
    save_statistics_to_pickle(statistics, DATA_DIR, name="statistics_lucas")

    # plot quiver 
    if return_flow:
        return flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Horn-Schunck Optical Flow')
    parser.add_argument('--dir', '-d', type=str, default='nasa', help='dir containing images')
    parser.add_argument('--save-name-colormap', type=str, default='lucas_output', help='Name of the saved colormap image')
    parser.add_argument('--window-min', type=int, default=5, help='Lower bound of window size parameter')
    parser.add_argument('--window-max', type=int, default=10, help='Upper bound of window size parameter')
    parser.add_argument('--window-step', type=int, default=5, help='Step size of window parameter')
    parser.add_argument('--return-flow', type=bool, default=False, help='Get results of flow')
    parser.add_argument('--gaussian-kernel', type=bool, default=False, help='Whether to use a Gaussian kernel or not')

    args = parser.parse_args()

    main(args)
