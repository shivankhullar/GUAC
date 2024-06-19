"""
This file contains generic utility functions.

Author: Shivan Khullar
Date: June 2024
"""

import numpy as np
from generic_utils.fire_utils import *
from galaxy_utils.gal_utils import *


def get_cloud_quick_info(snapnum, nmin, vir, cloud_num, params):
    """
    This is a function to get the quantities of a cloud from the bound_nX_alphaY_snapnum.dat file.

    Inputs:
        path: the path to the data
        snapnum: the snapshot number
        nmin: the minimum number of particles in a cloud
        vir: the virial parameter
        cloud_num: the cloud number
        params: the parameters object
    
    Outputs:
        cloud_total_mass: the total mass of the cloud
        cloud_centre: the centre of the cloud
        cloud_reff: the effective radius of the cloud
        dist: the distance of the cloud from the galactic centre
    """
    path = params.path+params.cph_sub_dir #path+'CloudPhinderData/n{nmin}_alpha{vir}/'.format(nmin=nmin, vir=vir)
    filename = 'bound_{snap_num}_n{nmin}_alpha{vir}.dat'.format(snap_num = snapnum, nmin = nmin, vir=vir)
    datContent = [i.strip().split() for i in open(path+filename).readlines()]
    
    header = params.dat_file_header_size
    i = cloud_num
    cloud_total_mass = float(datContent[i+header][0])
    cloud_centre_x = float(datContent[i+header][1])
    cloud_centre_y = float(datContent[i+header][2])
    cloud_centre_z = float(datContent[i+header][3])
    cloud_reff = float(datContent[i+header][7])
    cloud_vir = float(datContent[i+header][10])
    
    cloud_centre = np.array([cloud_centre_x, cloud_centre_y, cloud_centre_z])
    gal_centre = get_galaxy_centre(params, snapnum)
    dist = np.sqrt((cloud_centre_x-gal_centre[0])**2+(cloud_centre_y-gal_centre[1])**2+\
                    (cloud_centre_z-gal_centre[2])**2)
    
    return cloud_total_mass, cloud_centre, cloud_reff, cloud_vir, dist






class CloudChain():
    """
    Class to find out a cloud chain stemming from the first cloud.
    """
    def __init__(self, cloud_num, snap_num, params):
        file_name = params.path+params.sub_dir+params.filename_prefix+params.frac_thresh\
                    +"_"+str(params.start_snap)+"_"+str(params.last_snap)+"_names"+".txt"
        my_file = open(file_name, "r")

        content_list = my_file.readlines()
        cloud_list_names = []
        for i in range (0, len(content_list)-1):             #The last line is just \n.
            #if 
            names = str.split(content_list[i], ', ')
            if names[-1]=='\n':
                names = names[:-1]

            cloud_list_names.append(names)

        self.cloud_list = []
        self.cloud_nums = []
        self.snap_nums = []
        search_key = get_cloud_name(cloud_num, params)+'Snap'+str(snap_num)
        self.search_key = search_key
        flag = 0
        for i in range(0, len(cloud_list_names)):
            if search_key in cloud_list_names[i]:
                print ('Search key', search_key, i)
                self.cloud_list = cloud_list_names[i]

                flag = 1
                break
        
        for cloud in self.cloud_list:
            self.cloud_nums.append(int(cloud.split('Snap')[0].split('Cloud')[1]))
            self.snap_nums.append(int(cloud.split('Snap')[1]))
        
        #if flag==0:
        #    search_key = get_cloud_name0(cloud_num, params)+'Snap'+str(snap_num)
        #    for i in range(0, len(cloud_list_names)):
        #        if search_key in cloud_list_names[i]:
        #            print ('Search key', search_key, i)
        #            self.cloud_list = cloud_list_names[i]
        #            flag = 1
        #            break
                    
        if flag==0:
            print ('Cloud not found :(')
            

