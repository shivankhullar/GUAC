"""
This file contains some utility functions for rotating coordinates.

Author: Shivan Khullar
Date: March 2026
"""

import numpy as np


def get_rotation_matrix(axis='x', angle_deg=90):
    """
    Get rotation matrix for rotating coordinates around a given axis.
    
    Parameters:
    -----------
    axis : str
        Axis to rotate around ('x', 'y', or 'z')
    angle_deg : float
        Rotation angle in degrees
    
    Returns:
    --------
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    if axis.lower() == 'x':
        # Rotation around x-axis
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    elif axis.lower() == 'y':
        # Rotation around y-axis
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    elif axis.lower() == 'z':
        # Rotation around z-axis
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def rotate_coordinates(coords, axis='x', angle_deg=90, center=None):
    """
    Rotate coordinates around a given axis.
    
    Parameters:
    -----------
    coords : np.ndarray
        Nx3 array of coordinates
    axis : str
        Axis to rotate around ('x', 'y', or 'z')
    angle_deg : float
        Rotation angle in degrees
    center : np.ndarray, optional
        Center point for rotation (if None, rotates around origin)
    
    Returns:
    --------
    rotated_coords : np.ndarray
        Rotated Nx3 coordinates
    """
    R = get_rotation_matrix(axis, angle_deg)
    
    if center is not None:
        # Translate to origin, rotate, translate back
        coords_centered = coords - center
        rotated = np.dot(coords_centered, R.T)
        return rotated + center
    else:
        return np.dot(coords, R.T)



