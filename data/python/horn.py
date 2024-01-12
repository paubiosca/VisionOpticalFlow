import cv2
import numpy as np
import os

from flowToColorLib import flow_to_color
from readFlowFile import read_flow_file
from middlebury import computeColor, readflo
from utils import angular_error, get_image_names, get_GT_optical_flow_file
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
    # Compute derivatives
    Ix, Iy, It = gradhorn(im1, im2)
    
    # Initialize flow vectors
    u = np.zeros(im1.shape)
    v = np.zeros(im2.shape)
    
    # Kernel for the convolution as per the new instructions
    A = np.array([[1/12, 1/6, 1/12], 
                  [1/6,  0,   1/6 ], 
                  [1/12, 1/6, 1/12]])
    
    # Iterate to refine the estimates
    for _ in range(N):
        # Compute u_avg and v_avg by convolving u and v with kernel A
        u_avg = cv2.filter2D(u, -1, A)
        v_avg = cv2.filter2D(v, -1, A)
        
        # Compute the update for u and v as per the new instructions
        u = u_avg - Ix * (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        
    return (u, v)


def read_image_placeholder(filename):
    # Read the image
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Convert to float
    im = np.float32(im) / 255.0
    
    return im

def main(folder = 'nasa', method = 'horn'):
    # Get the current working directory
    cwd = os.getcwd()
    SCRIPTS_DIR = os.path.dirname(cwd)
    DATA_DIR = os.path.join(SCRIPTS_DIR, 'data', folder)
    print('Data directory: ', DATA_DIR)
    
    image_name_1, image_name_2 = get_image_names(folder)
    
    I1 = read_image_placeholder(os.path.join(DATA_DIR, image_name_1))
    I2 = read_image_placeholder(os.path.join(DATA_DIR, image_name_2))

    # Set parameters for the algorithm
    alpha = 1  # the regularization parameter
    N = 10000   # number of iterations

    # Apply the Horn-Schunck method
    u, v = horn(I1, I2, alpha, N)
    FLOW = np.dstack((u, v))
    
    # Use FlowToColor to visualize the optical flow
    flow_color_image = computeColor(np.dstack((u, v)))
    # Display image with optical flow on top
    print('Saving output to data/horn_output_2.png')
    cv2.imwrite(os.path.join(DATA_DIR, 'horn_output_2.png'), flow_color_image)
    
    # LOAD GROUND TRUTH FLOW TO BE DONE!!!
    GT_FLOW = get_GT_optical_flow_file(DATA_DIR)
    
    if (GT_FLOW is not None):
        # Compute the angular error
        mean_angle_error, std_angle_error = angular_error(FLOW, GT_FLOW)
        print('Mean angular error: {:.2f}'.format(round(mean_angle_error, 2)))
        print('Standard deviation of angular error: {:.2f}'.format(round(std_angle_error, 2)))
    
    
if __name__ == '__main__':
    FOLDER = 'nasa'
    main(folder = FOLDER)
