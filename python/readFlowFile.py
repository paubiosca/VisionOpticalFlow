import numpy as np
import struct

def read_flow_file(filename):
    """
    Read optical flow from file in .flo format.
    
    Parameters:
    - filename: string, path to the .flo file
    
    Returns:
    - img: numpy.ndarray, the optical flow in 2-band image format
    """
    TAG_FLOAT = 202021.25  # the tag for .flo file header

    # Check for correct file extension
    if not filename.endswith('.flo'):
        raise ValueError('readFlowFile: filename {} should have extension .flo'.format(filename))

    with open(filename, 'rb') as f:
        # Read and check the header
        tag = struct.unpack('<f', f.read(4))[0]
        if tag != TAG_FLOAT:
            raise ValueError('readFlowFile: wrong tag (possibly due to big-endian machine?)')

        # Read the width and height
        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]

        # Sanity check for width and height
        if width < 1 or width > 99999:
            raise ValueError('readFlowFile: illegal width {}'.format(width))
        if height < 1 or height > 99999:
            raise ValueError('readFlowFile: illegal height {}'.format(height))

        # Read the flow data
        nBands = 2
        nb_pixels = width * height
        tmp = np.fromfile(f, np.float32, nb_pixels * nBands)
        img = np.zeros((height, width, nBands), dtype=np.float32)

        # Reshape the flow data into image format
        img[:, :, 0] = tmp[0:nb_pixels * nBands:2].reshape((height, width))
        img[:, :, 1] = tmp[1:nb_pixels * nBands:2].reshape((height, width))

    return img