import cv2
import numpy as np
import os

from flowToColorLib import flow_to_color
from readFlowFile import read_flow_file

def grad_horn(im1, im2):
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
    Ix, Iy, It = grad_horn(im1, im2)
    
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


def main():
    # Get the current working directory
    cwd = os.getcwd()
    SCRIPTS_DIR = os.path.dirname(cwd)
    DATA_DIR = os.path.join(SCRIPTS_DIR, 'data', 'nasa')
    print('Data directory: ', DATA_DIR)
    
    I1 = read_image_placeholder(os.path.join(DATA_DIR, 'nasa9.png'))
    I2 = read_image_placeholder(os.path.join(DATA_DIR, 'nasa10.png'))

    # Set parameters for the algorithm
    alpha = 0.8  # the regularization parameter
    N = 10000    # number of iterations

    # Apply the Horn-Schunck method
    u, v = horn(I1, I2, alpha, N)
    
    # Use FlowToColor to visualize the optical flow
    flow_color_image = flow_to_color(np.dstack((u, v)))
    # Display image with optical flow on top
    print('Saving output to data/horn_output.png')
    cv2.imwrite(os.path.join(DATA_DIR, 'horn_output.png'), flow_color_image)
    
    # LOAD GROUND TRUTH FLOW TO BE DONE!!!
    # # Test the function with a placeholder filename (this will only work with a valid .flo file)
    # filename = os.path.join(DATA_DIR, 'nasa_horn_png')
    # flow_img_GT = read_flow_file(filename)
    # print(flow_img_GT.shape)  # Should show (height, width, 2) if successful
    
if __name__ == '__main__':
    main()
