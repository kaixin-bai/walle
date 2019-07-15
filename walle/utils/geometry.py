import numpy as np


def estimate_rigid_transform_rotm(X, Y):
    """Determines the rotation and translation that best aligns X to Y.

    Args:
        X: a numpy array of shape (N, 3).
        Y: a numpy array of shape (N, 3).

    Returns:
        rot: a numpy array of shape (3, 3).
        trans: a numpy array of shape (3,).
    """
    # find centroids
    X_c = np.mean(X, axis=0)
    Y_c = np.mean(Y, axis=0)

    # shift
    X_s = X - X_c
    Y_s = Y - Y_c

    # compute SVD of covariance matrix
    cov = Y_s.T @ X_s
    u, _, vt = np.linalg.svd(cov)

    # determine rotation
    rot = u @ vt
    if np.linalg.det(rot) < 0.:
        logging.warn("Special reflection case.")
        vt[2, :] *= -1
        rot = u @ vt

    # determine optimal translation
    trans = Y_c - rot @ X_c

    return rot, trans


def estimate_rigid_transform_quat(X, Y):
    pass