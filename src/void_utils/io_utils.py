"""
This file contains utility functions to get galaxy quantities.

Author: Shivan Khullar
Date: March 2025
"""

from generic_utils.fire_utils import *
import generic_utils.constants as const
from void_utils.calculate_void_quants import *



import glob
import numpy as np
#import yt
import h5py
from meshoid import Meshoid
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import colorcet as cc
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from scipy import interpolate 
#import pandas as pd

from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial import cKDTree

#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
from scipy import ndimage
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import measure
from tqdm import tqdm
import pickle
import os




def read_linked_voids_file(filename):
    list_of_lists = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace and split by commas
            entries = [entry.strip() for entry in line.strip().split(',') if entry.strip()]
            if entries:
                list_of_lists.append(entries)
    return list_of_lists





def get_all_voids_data_hdf5(snap_num, params, contour_only=False):
    """
    This is a function to get the quantities of a void from the hdf5 file.
    Inputs:
    void_num: int
    snap_num: int
    params: Params object

    Outputs:
    void_data: dict
    """
    hdf5_file_name = params.path+params.sub_dir+'/'+params.phinder_sub_dir+params.hdf5_file_prefix+str(snap_num)+'.hdf5'

    f = h5py.File(hdf5_file_name, 'r')
    contour_list = []
    void_data_list = []
    for key in f.keys():
        #if key.startswith('Void'):
        void_name = key
        #    break

        #void_name = get_cloud_name(void_num, params)
        g = f[void_name]
        
        void_data = {}
        void_data['Name'] = void_name
        void_data['Contour'] = np.array(g['Contour'][:])

        
        if contour_only:
            contour_list.append(void_data['Contour'])
            #return void_data
        
        else:
            h = g['PartType0']        
            void_data['PartType0'] = {}
            void_data['PartType0']['Coordinates'] = np.array(h['Coordinates'])
            void_data['PartType0']['Velocities'] = np.array(h['Velocities'])
            void_data['PartType0']['Masses'] = np.array(h['Masses'])
            void_data['PartType0']['SmoothingLength'] = np.array(h['SmoothingLength'])
            void_data['PartType0']['Density'] = np.array(h['Density'])
            void_data['PartType0']['Temperature'] = np.array(h['Temperature'])
            void_data['PartType0']['Pressure'] = np.array(h['Pressure'])
            void_data['PartType0']['ParticleIDs'] = np.array(h['ParticleIDs'])
            void_data['PartType0']['ParticleIDGenerationNumber'] = np.array(h['ParticleIDGenerationNumber'])
            void_data['PartType0']['ParticleChildIDsNumber'] = np.array(h['ParticleChildIDsNumber'])
            #return void_data
            void_data_list.append(void_data)
        
    if contour_only:
        return contour_list
    else:
        return void_data_list


def get_void_data_hdf5(void_num, snap_num, params, contour_only=False):
    """
    This is a function to get the quantities of a void from the hdf5 file.
    Inputs:
    void_num: int
    snap_num: int
    params: Params object

    Outputs:
    void_data: dict
    """
    hdf5_file_name = params.path+params.sub_dir+'/'+params.phinder_sub_dir+params.hdf5_file_prefix+str(snap_num)+'.hdf5'

    f = h5py.File(hdf5_file_name, 'r')
    void_name = get_cloud_name(void_num, params)
    g = f[void_name]
    
    void_data = {}
    void_data['Name'] = void_name
    void_data['Contour'] = np.array(g['Contour'][:])

    if contour_only:
        return void_data
    
    else:
        h = g['PartType0']        
        void_data['PartType0'] = {}
        void_data['PartType0']['Coordinates'] = np.array(h['Coordinates'])
        void_data['PartType0']['Velocities'] = np.array(h['Velocities'])
        void_data['PartType0']['Masses'] = np.array(h['Masses'])
        void_data['PartType0']['SmoothingLength'] = np.array(h['SmoothingLength'])
        void_data['PartType0']['Density'] = np.array(h['Density'])
        void_data['PartType0']['Temperature'] = np.array(h['Temperature'])
        void_data['PartType0']['Pressure'] = np.array(h['Pressure'])
        void_data['PartType0']['ParticleIDs'] = np.array(h['ParticleIDs'])
        void_data['PartType0']['ParticleIDGenerationNumber'] = np.array(h['ParticleIDGenerationNumber'])
        void_data['PartType0']['ParticleChildIDsNumber'] = np.array(h['ParticleChildIDsNumber'])
        return void_data
    
    












class VoidChain():
    """
    Class to find out a cloud chain stemming from the first cloud.
    """
    def __init__(self, cloud_num, snap_num, params):
        #file_name = params.path+params.sub_dir+params.filename_prefix+params.frac_thresh\
        #            +"_"+str(params.start_snap)+"_"+str(params.last_snap)+"_names"+".txt"
        file_name = params.linked_filename
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
        search_key = get_cloud_name(cloud_num, params)+params.snapshot_prefix+str(snap_num)
        self.search_key = search_key
        flag = 0
        for i in range(0, len(cloud_list_names)):
            if search_key in cloud_list_names[i]:
                print ('Search key', search_key, i)
                self.cloud_list = cloud_list_names[i]

                flag = 1
                break
        
        for cloud in self.cloud_list:
            self.cloud_nums.append(int(cloud.split(params.snapshot_prefix)[0].split(params.cloud_prefix)[1]))
            self.snap_nums.append(int(cloud.split(params.snapshot_prefix)[1]))
        
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
            


def load_linked_void_data(params, lifetime_cutoff=10, run_new_calculation=True, void_list=None):
    """
    Load linked void data from a pickle file.
    Parameters:
        params: Params object containing path and other parameters.
        lifetime_cutoff: int, optional, default=0
            The lifetime cutoff for the linked voids to be loaded.
    Returns:
        linked_void_data_list: list
            A list of linked void data dictionaries.
    """
    output_dir = os.path.join(params.path, "linked_void_data")
    filename = f"linked_void_data_list_{lifetime_cutoff}.pkl"
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "rb") as f:
            linked_void_data_list = pickle.load(f)
        print(f"Loaded {len(linked_void_data_list)} linked voids from {filepath}.")
    except FileNotFoundError:
        print (f"File {filepath} not found. Running new calculation instead.")
        if run_new_calculation and void_list is not None:
            print ("Computing linked void data...")
            linked_void_data_list = compute_linked_void_data_list(params, void_list, lifetime_cutoff)
            print(f"New linked void data calculated and saved to {filepath}.")
            linked_void_data_list = load_linked_void_data(params, lifetime_cutoff=lifetime_cutoff, run_new_calculation=False, void_list=void_list)
        else:
            linked_void_data_list = []
            print(f"Returning an empty list, because no new calculation was requested. Maybe provide the void list")
        #print(f"File {filepath} not found. Returning an empty list.")
        #linked_void_data_list = []
    
    return linked_void_data_list
