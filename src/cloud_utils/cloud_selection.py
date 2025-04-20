"""
This file contains functions to select clouds from a population in a single snapshot, based on some critera for selection.

Author: Shivan Khullar
Date: June 2024
"""

import numpy as np
from cloud_utils.calculate_cloud_quants import *

class CloudSelector():
    """
    Class to select clouds based on some ranges in their properties.
    """
    def __init__(self, cloud_mass_range=[1e5,1e8], cloud_reff_range=[0,1000], cloud_hmrad_range=None,\
                 cloud_r_gal_range=[1,25], min_cloud_num_part=32, cloud_in_galaxy=False):

        self.cloud_mass_range = cloud_mass_range        # in Msun
        self.cloud_reff_range = cloud_reff_range        # in pc
        self.cloud_r_gal_range = cloud_r_gal_range      # in kpc
        self.min_cloud_num_part = min_cloud_num_part    # Minimum number of particles in a cloud.
        self.cloud_in_galaxy = cloud_in_galaxy          # If True, only clouds within the galaxy are selected.
        self.cloud_hmrad_range = cloud_hmrad_range      # in pc

       
    def get_cloud_inds(self, cloud_data, params):
        """
        Function to select clouds based on some ranges in their properties. 

        Inputs:
            cloud_mass_range: A list of two elements, lower and upper limits of the cloud mass.
            cloud_reff_range: A list of two elements, lower and upper limits of the cloud effective radius.
            cloud_hmrad_range: A list of two elements, lower and upper limits of the cloud half-mass radius.
            cloud_r_gal_range: A list of two elements, lower and upper limits of the distance of the cloud from the galactic center.
            min_cloud_num_part: Minimum number of particles in the cloud.
            milky_way: Whether to select clouds in the disk or not.

        Outputs:
            cloud_inds: Indices of the clouds that satisfy the conditions.                
        """

        self.cloud_inds = []
        inds_list = []
        if self.cloud_mass_range is not None:
            inds_list.append(np.where((cloud_data.cloud_total_masses*1e10>=self.cloud_mass_range[0]) & \
                            (cloud_data.cloud_total_masses*1e10<=self.cloud_mass_range[1]))[0])
        if self.cloud_reff_range is not None:
            inds_list.append(np.where((cloud_data.cloud_reffs*1e3>=self.cloud_reff_range[0]) & \
                            (cloud_data.cloud_reffs*1e3<=self.cloud_reff_range[1]))[0])
        if self.cloud_hmrad_range is not None:
            inds_list.append(np.where((cloud_data.cloud_hmrads*1e3>=self.cloud_hmrad_range[0]) & \
                            (cloud_data.cloud_hmrads<=self.cloud_hmrad_range[1]))[0])
        if self.cloud_r_gal_range is not None:
            inds_list.append(np.where((cloud_data.r_gal_dists>=self.cloud_r_gal_range[0]) & \
                            (cloud_data.r_gal_dists<=self.cloud_r_gal_range[1]))[0])
        if self.min_cloud_num_part is not None:
            inds_list.append(np.where(cloud_data.cloud_num_parts>=self.min_cloud_num_part)[0])
        if self.cloud_in_galaxy:
            inds_list.append(np.where((np.abs(cloud_data.z_dists)<=params.h) & \
                            (cloud_data.r_gal_dists<=params.r_gal))[0])
        
        # Take the intersection of the indices.
        self.cloud_inds = inds_list[0]
        for i in range(1, len(inds_list)):
            self.cloud_inds = np.intersect1d(self.cloud_inds, inds_list[i])
        
        # Return the cloud_nums corresponding to the selected clouds inds
        #self.cloud_nums = cloud_data.cloud_nums[self.cloud_inds]
        #return self.cloud_nums

        return self.cloud_inds




