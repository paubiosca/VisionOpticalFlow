import numpy as np
import cv2

def compute_color(u, v):
    """
    Given a normalized optical flow field (u, v), compute the corresponding RGB image.
    """
    # Color encoding of flow vectors
    n = 8
    colorwheel = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 1, 1], [0, 0, 1], [1, 0, 1]])
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(int)

    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((u.shape[0], u.shape[1], 3))
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] *= 0.75  # out of range

        img[:, :, i] = np.floor(255 * col * (1 - idx) + 255 * col * idx).astype(np.uint8)

    return img

def flow_to_color(flow, max_flow=None):
    """
    Visualize optical flow field.
    :param flow: Optical flow field with shape (height, width, 2).
    :param max_flow: Max flow to map flow vectors to color.
    :return: Color image representation of the flow field.
    """
    UNKNOWN_FLOW_THRESH = 1e9
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Mask to identify unknown flows
    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idx_unknown] = 0
    v[idx_unknown] = 0

    # Find max flow to normalize flow field
    rad = np.sqrt(u**2 + v**2)
    maxrad = rad.max()

    if max_flow is not None:
        maxrad = max_flow

    # Normalize the flow field
    u /= maxrad + np.finfo(float).eps
    v /= maxrad + np.finfo(float).eps

    # Compute color representation
    img = compute_color(u, v)

    # Set unknown flow pixels to 0 (black)
    img[idx_unknown] = 0

    return img
