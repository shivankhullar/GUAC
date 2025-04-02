"""
This file contains utility functions to get galaxy quantities.

Author: Shivan Khullar
Date: April 2025
"""

from generic_utils.fire_utils import *
import generic_utils.constants as const
from generic_utils.load_fire_snap import *
from galaxy_utils.gal_utils import *
from void_utils.io_utils import *




def contour_centroid(contour):
    """
    Compute the centroid of a 2D polygonal contour.
    
    Parameters:
        contour (ndarray): Nx2 array of (x, y) points, assumed to be a closed loop.
        
    Returns:
        (cx, cy): Tuple representing centroid coordinates.
    """
    x = contour[:, 0]
    y = contour[:, 1]

    # Ensure closed contour
    if not np.allclose(contour[0], contour[-1]):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Shoelace area
    A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

    # Centroid formula
    cx = (1/(6*A)) * np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
    cy = (1/(6*A)) * np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))

    return np.array([cx, cy])




def get_linked_void_quants(params, void_list):
    #count = 0
    linked_void = {}
    linked_void['Name'] = void_list[0]
    #linked_void_name = void_list[0]
    for void in void_list:
        snap_num = int(void.split(params.snapshot_prefix)[1])
        void_num = int(void.split(params.cloud_prefix)[1].split(params.snapshot_prefix)[0])
        print ('Snapshot:', snap_num, 'Void:', void_num)
        gal_quants0 = load_gal_quants(params, snap_num)
        center = gal_quants0.gal_centre_proj
        #smooth_fac = 1
        
        #M = Meshoid(gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], gal_quants0.data["SmoothingLength"])

        void_data = get_void_data_hdf5(void_num, snap_num, params, contour_only=True)
        contour = void_data['Contour']
        void_max_x, void_max_y = np.max(contour[:, 0]), np.max(contour[:, 1])
        void_min_x, void_min_y = np.min(contour[:, 0]), np.min(contour[:, 1])

        void_center = np.array([void_min_x + (void_max_x - void_min_x)/2, void_min_y + (void_max_y - void_min_y)/2])
        #define a box size around the void contour
        box_size = max(void_max_x - void_min_x, void_max_y - void_min_y)

        if box_size < 2:
            box_size = 2

        box_size_multiplier = 2
        box_size *= box_size_multiplier
