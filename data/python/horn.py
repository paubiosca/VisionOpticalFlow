import argparse
import cv2
import numpy as np
import os

from flowToColorLib import flow_to_color
from readFlowFile import read_flow_file
from middlebury import computeColor, readflo
from utils import angular_error, get_image_names, get_GT_optical_flow_file, end_point_error, print_statistics, relative_norm_error, save_statistics_to_csv
import os

def gradhorn(im1, im2):
    # Compute derivatives
    Ix = 0.25 * (np.roll(im1, -1, axis=1) - np.roll(im1, 1, axis=1) + 
                 np.roll(im2, -1, axis=1) - np.roll(im2, 1, axis=1))
    Iy = 0.25 * (np.roll(im1, -1, axis=0) - np.roll(im1, 1, axis=0) + 
                 np.roll(im2, -1, axis=0) - np.roll(im2, 1, axis=0))
    It = 0.25 * (np.roll(im2, -1, axis=1) - np.roll(im1, -1, axis=1) + 
                 np.roll(im2, -1, axis=0) - np.roll(im1, -1, axis=0) +
                 np.roll(im2, -1, axis=(0,1)) - np.roll(im1, -1, axis=(0,1)))
    return (Ix, Iy, It)

def horn(im1, im2, alpha, N):
    # 2. Compute Ix, Iy et It (the derivatives in the x, y and t directions)
    Ix, Iy, It = gradhorn(im1, im2)
    
    # Initialize flow vectors:
    # 4. u0 = v0 = 0, two images having same size than I1
    u = np.zeros(im1.shape)
    v = np.zeros(im2.shape)
    
    # Kernel for the convolution as per the new instructions
    A = np.array([[1/12, 1/6, 1/12], 
                  [1/6,  0,   1/6 ], 
                  [1/12, 1/6, 1/12]])
    
    # Iterate to refine the estimates
    # 5. For k = 0 to N − 1:
    for _ in range(N):
        # (a) Compute u_avg and v_avg by convolving u and v with kernel A
        u_avg = cv2.filter2D(u, -1, A)
        v_avg = cv2.filter2D(v, -1, A)
        
        # (b) Compute the update for u and v as per the new instructions
        u = u_avg - Ix * (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        
    return (u, v)


def read_image_placeholder(filename):
    # Read the image
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Convert to float
    im = np.float32(im) / 255.0
    
    return im

def main(args):
    # 3. Choose N (number of iterations) and α (regularization) --> parameters of the algorithm
    dir = args.dir
    alpha = args.alpha  # the regularization parameter
    N = args.iterations   # number of iterations

    # Get the current working directory
    cwd = os.getcwd()
    SCRIPTS_DIR = os.path.dirname(cwd)
    DATA_DIR = os.path.join(SCRIPTS_DIR, 'data', dir)
    print('Data directory: ', DATA_DIR)
    
    # 1. Read two images I1, I2 (same size and dimensions)
    image_name_1, image_name_2 = get_image_names(dir)
    I1 = read_image_placeholder(os.path.join(DATA_DIR, image_name_1))
    I2 = read_image_placeholder(os.path.join(DATA_DIR, image_name_2))


    # Apply the Horn-Schunck method
    u, v = horn(I1, I2, alpha, N)
    flow = np.dstack((u, v)) # estimated flow
    
    # 6. Visualization of optical flow (velocity map) with function computeColor() from middlebury.py
    flow_color_image = computeColor(np.dstack((u, v)))
    # Display image with optical flow on top
    print('Saving output to data/horn_output_2.png')
    cv2.imwrite(os.path.join(DATA_DIR, 'horn_output_2.png'), flow_color_image)
    
    # LOAD GROUND TRUTH FLOW TO BE DONE!!!
    # 7. If available: read the ground truth with the function readFlowFile() and compare with (uN , vN ) (see next subsection).
    gt_flow = get_GT_optical_flow_file(DATA_DIR)


    # 3. if a ground truth is available, you would compute several statistics (mean, standard deviation) of End-Point error, angular error, norm error. 
    statistics = {}
    if (gt_flow is not None):
        # Compute several errors
        error_functions = [ angular_error, end_point_error, relative_norm_error]

        for error_func in error_functions:
            error_img = error_func(flow, gt_flow)
            mean_error = np.mean(error_img)
            std_error = np.std(error_img)

            # Initialize the dictionary if it doesn't exist
            statistics[error_func.__name__] = {"mean": [], "std": []}
            
            statistics[error_func.__name__]["mean"].append(mean_error)
            statistics[error_func.__name__]["std"].append(std_error)
        
        # Display the statistics
        print_statistics(statistics)

        # save statistics into csv file
        save_statistics_to_csv(statistics, DATA_DIR)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Horn-Schunck Optical Flow')
    parser.add_argument('--dir', '-d', type=str, default='nasa', help='dir containing images')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    
    args = parser.parse_args()
    main(args)
