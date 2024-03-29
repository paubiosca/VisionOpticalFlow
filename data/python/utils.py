import numpy as np
import os
from middlebury import computeColor, readflo

def angular_error(estimated_flow, ground_truth_flow):
    """
    Compute the angular error between the estimated flow and the ground truth flow.

    Parameters:
    estimated_flow: numpy.ndarray, the estimated optical flow.
    ground_truth_flow: numpy.ndarray, the ground truth optical flow.

    Returns:
    angular_error: numpy.ndarray, the angular errors for each pixel.
    """
    # Dot product between estimated flow and ground truth flow
    dot_product = np.sum(estimated_flow * ground_truth_flow, axis=2)
    # Magnitudes of estimated flow and ground truth flow
    mag_estimated = np.sqrt(np.sum(estimated_flow ** 2, axis=2))
    mag_ground_truth = np.sqrt(np.sum(ground_truth_flow ** 2, axis=2))

    # Compute the cosine of the angular error
    cos_angle_error = (1 + dot_product) / (np.sqrt(1 + mag_estimated ** 2) * np.sqrt(1 + mag_ground_truth ** 2))
    # Ensure the values are in the valid range [-1, 1] for arccos
    cos_angle_error = np.clip(cos_angle_error, -1, 1)
    
    # Compute the angular error in radians
    angle_error = np.arccos(cos_angle_error)
    
    # Convert to degrees
    angle_error_degrees = np.degrees(angle_error)
    
    # Calculate statistics
    mean_angular_error = np.mean(angle_error_degrees)
    std_angular_error = np.std(angle_error_degrees)

    return mean_angular_error, std_angular_error

def get_image_names(folder = 'nasa'):
    if folder == 'nasa':
        image_name_1 = 'nasa9.png'
        image_name_2 = 'nasa10.png'
    elif folder == 'square':
        image_name_1 = 'square9.png'
        image_name_2 = 'square10.png'
    elif folder == 'rubic':
        image_name_1 = 'rubic9.png'
        image_name_2 = 'rubic10.png'
    elif folder == 'rubberwhale':
        image_name_1 = 'frame10.png'
        image_name_2 = 'frame11.png'
    elif folder == 'mysine':
        image_name_1 = 'mysine9.png'
        image_name_2 = 'mysine10.png'
    elif folder == 'taxi':
        image_name_1 = 'taxi9.png'
        image_name_2 = 'taxi10.png'
    elif folder == 'yosemite':
        image_name_1 = 'yos9.png'
        image_name_2 = 'yos10.png'
    
    return image_name_1, image_name_2

def get_GT_optical_flow_file(DATA_DIR):
    # Get the current working directory
    # Check if in data dir there is a file ending with .flo
    file_bool = False
    file_name = ''
    for file in os.listdir(DATA_DIR):
        if file.endswith(".flo"):
            file_bool = True
            file_name = file
            
    if file_bool == False:
        print('No .flo file found in data dir')
        return None
    else:
        print('Found .flo file: ', file_name)
        gt_flow = readflo(os.path.join(DATA_DIR, file_name))
        return gt_flow
            
    
        
        
    
    
        