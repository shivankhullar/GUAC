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



from tqdm import tqdm



def get_contour_centroid(contour):
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


def get_effective_radius(contour):
    """
    Compute the effective radius of a 2D polygonal contour assuming a circular shape.
    """
    x, y = contour[:, 1], contour[:, 0]
    # Compute area using the Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    # Compute effective radius assuming a circular shape
    effective_radius = np.sqrt(area / np.pi)
    return effective_radius


def get_linked_void_quants(params, void_list):
    from void_utils.io_utils import get_void_data_hdf5
    #count = 0
    linked_void = {}
    linked_void['name'] = void_list[0]
    #linked_void_name = void_list[0]
    effective_radius = []
    void_centers = []
    for void in void_list:
        snap_num = int(void.split(params.snapshot_prefix)[1])
        void_num = int(void.split(params.cloud_prefix)[1].split(params.snapshot_prefix)[0])
        #print ('Snapshot:', snap_num, 'Void:', void_num)
        #gal_quants0 = load_gal_quants(params, snap_num)
        #center = gal_quants0.gal_centre_proj
        #smooth_fac = 1
        
        #M = Meshoid(gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], gal_quants0.data["SmoothingLength"])

        void_data = get_void_data_hdf5(void_num, snap_num, params, contour_only=True)
        contour = void_data['Contour']
        void_center = get_contour_centroid(contour)
        void_centers.append(void_center)
        effective_radius.append(get_effective_radius(contour))
    
    void_centers = np.array(void_centers)
    effective_radius = np.array(effective_radius)
    linked_void['effective_radius'] = np.median(effective_radius)
    linked_void['radius_time'] = effective_radius
    linked_void['center_time'] = void_centers
    linked_void['maximum_radius'] = np.max(effective_radius)

    return linked_void


# We will apply some selection criteria to the linked voids to identify the ones where the radius is increasing over time.
def select_voids(linked_void_data_list):
    selected_voids = []
    for i in range (0, len(linked_void_data_list)):# in linked_void_data_list:
        void_data = linked_void_data_list[i]
        radius_time = void_data['radius_time']
        if len(radius_time) < 20:
            continue
        # Check if the radius is increasing over time
        # Using a loose criterion first
        if radius_time[-1] < 1.5 * radius_time[0]:
            continue
        
        # Avoid voids that have a decreasing radius at any point in time
        #if np.any(np.diff(radius_time) < 0):
        #    continue

        # Avoid voids that have a large increase in radius between consecutive snapshots
        if np.any(np.diff(radius_time) > 0.5 * radius_time[:-1]):
            continue

        else:
            selected_voids.append(void_data)

        # This is a strict criterion
        #if np.all(np.diff(radius_time) > 0):
        #    selected_voids.append(void_data)
    return selected_voids



def compute_linked_void_data_list(params, linked_voids_list, lifetime_cutoff, save_file=True):
    """
    Compute the linked void data for a list of voids.
    Parameters:
        params: Parameters object containing simulation parameters.
        void_list: List of void names to process.
        lifetime_cutoff: Minimum number of snapshots a void must be present in to be considered.
        save_file: Whether to save the computed data to a file.
    Returns:
        linked_void_data_list: List of dictionaries containing linked void data.
    """

    linked_void_data_list = []
    for void_list in tqdm(linked_voids_list, desc="Processing linked voids"):
        linked_void_data_list.append(get_linked_void_quants(params, void_list))


    if save_file:
        output_dir = os.path.join(params.path, "linked_void_data")
        os.makedirs(output_dir, exist_ok=True)

        # Save the data
        filename = f"linked_void_data_list_{lifetime_cutoff}.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(linked_void_data_list, f)
        print(f"Linked void data saved successfully to {filepath}.")
    else:
        print("Linked void data not saved to file.")

    return linked_void_data_list




#def calculate_SNe_within_void_countours():
