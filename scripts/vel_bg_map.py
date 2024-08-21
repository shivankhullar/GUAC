#!/usr/bin/env python
"""
vel_bg_map.py: "Create initial conditions file for FIRE+STARFORGE simulation 
                            with a refinement particle placed at a custom position.
                            The custom position is specified by the user. The script will
                            then choose particles within a certain distance of the custom
                            position to calculate the center of mass velocity of the SMBH particle.
                            One can specify if the particles should follow stars or gas particles while
                            deciding the original position for the refinement particle.
                            The script will then create an hdf5 file with the ICs."

Usage: vel_bg_map.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: ./]
    --save_path=<save_path>                             Path to save the images [default: ./]
    --image_box_size=<image_box_size>                   Size of the image box [default: 15]
    --snapnum_range=<snapnum_range>                     Range of snapshots to plot [default: 0,100]
    --res=<res>                                         Resolution of the image [default: 2048]
"""


from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from generic_utils.load_fire_snap import *
from galaxy_utils.gal_utils import *
import time



import glob
#import yt
import h5py
from meshoid import Meshoid
#matplotlib.use('Agg')
from matplotlib import colors
import colorcet as cc
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from scipy import interpolate 


import numpy as np
import multiprocessing as mp
from numba import njit, prange, jit
from docopt import docopt

def save_data(save_path, name, data, snapnum):
    filename = save_path + name + "_snap" + str(snapnum) + ".npy"
    np.save(filename, data)

#@njit(parallel=False)

@njit(nopython=True)
def compute_com_vel(coords, vels, masses, cut_off_distance):
    num_particles = len(coords)
    vels_sub_1x = np.zeros_like(vels)
    for i in range(num_particles):
        pos = coords[i]
        dist = np.empty(num_particles)
        total_mass = 0
        mass_vel_x = 0
        mass_vel_y = 0
        mass_vel_z = 0
        
        for j in range(num_particles):
            x_dist = coords[j][0] - pos[0]
            x_dist = x_dist**2
            y_dist = coords[j][1] - pos[1]
            y_dist = y_dist**2
            z_dist = coords[j][2] - pos[2]
            z_dist = z_dist**2
            dist[j] = np.sqrt(x_dist + y_dist + z_dist)
            
            if dist[j] < cut_off_distance:
                total_mass += masses[j]
                mass_vel_x += masses[j]*vels[j, 0]
                mass_vel_y += masses[j]*vels[j, 1]
                mass_vel_z += masses[j]*vels[j, 2]
        
        vels_sub_1x[i, 0] -= mass_vel_x/total_mass
        vels_sub_1x[i, 1] -= mass_vel_y/total_mass
        vels_sub_1x[i, 2] -= mass_vel_z/total_mass

    return vels_sub_1x




def process_snapshot(params, snapnum, cut_off_distance):
    #Start timer
    start = time.time()

    print("Processing snapshot:", snapnum)
    snapdir = params.path
    print ("snapdir: ", snapdir)
    masses = load_fire_data("Masses", 0, snapdir, snapnum)
    coords = load_fire_data("Coordinates", 0, snapdir, snapnum)
    hsml = load_fire_data("SmoothingLength", 0, snapdir, snapnum)
    vels = load_fire_data("Velocities", 0, snapdir, snapnum)

    
    vels_sub_1x = compute_com_vel(coords, vels, masses, cut_off_distance)
    print ("1x done in ", time.time() - start)
    vels_sub_5x = compute_com_vel(coords, vels, masses, cut_off_distance * 5)
    print ("5x done in ", time.time() - start)
    vels_sub_0_5x = compute_com_vel(coords, vels, masses, cut_off_distance * 0.5)
    print ("0.5x done in ", time.time() - start)
    vels_sub_0_1x = compute_com_vel(coords, vels, masses, cut_off_distance * 0.1)
    print ("0.1x done in ", time.time() - start)

    save_data(params.save_path, "vels_sub_1x", vels_sub_1x, snapnum)
    save_data(params.save_path, "vels_sub_5x", vels_sub_5x, snapnum)
    save_data(params.save_path, "vels_sub_0_5x", vels_sub_0_5x, snapnum)
    save_data(params.save_path, "vels_sub_0_1x", vels_sub_0_1x, snapnum)

    print("Time taken for this snapshot:", time.time() - start)
    
    del masses, coords, hsml, vels






if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    save_path = path + sim + '/' + args['--save_path']
    image_box_size = float(args['--image_box_size'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    cut_off_distance = 1.0

    start_snap = snapnum_range[0]
    last_snap = snapnum_range[1]
    #field_names = ["names", "masses"]
    filename_prefix = "Linked_Clouds_"

    #No. of digits in the names of the clouds and the snapshots.
    cloud_num_digits = 4
    snapshot_num_digits = 4

    cloud_prefix = "Cloud"
    image_path = 'img_data/'
    image_filename_prefix = 'center_proj_'
    image_filename_suffix = '.hdf5'
    hdf5_file_prefix = 'Clouds_'


    age_cut = 1
    dat_file_header_size=8
    snapshot_prefix="Snap"
    star_data_sub_dir = "StarData/"
    gas_data_sub_dir = "GasData/"
    cph_sub_dir="CloudPhinderData/"
    #cph_sub_dir='m12i_restart/'
    frac_thresh='thresh0.3'

    r_gal = 25
    h = 3 #0.4

    #save_path = './data/'
    #snapnum = 650

    nmin = 10
    vir = 5
    #sim = "m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5"
    sim = args['--sim']
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
    filename = path+sub_dir+filename_prefix+str(start_snap)+"_"+str(last_snap)+"names"+".txt"


    
    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)
        

    #with mp.Pool(mp.cpu_count()) as pool:
    #    pool.starmap(
    #        process_snapshot, [(params, snapnum, cut_off_distance) for snapnum in range(snapnum_range[0], snapnum_range[1] + 1)]
    #    )

    print("Compiling the function")
    coords = np.random.rand(100,3)
    vels = np.random.rand(100,3)
    masses = np.random.rand(100)
    compute_com_vel(coords, vels, masses, 0.1)
    print("Function compiled")

    for snapnum in range(snapnum_range[0], snapnum_range[1] + 1):
        process_snapshot(params, snapnum, cut_off_distance)


"""
# 5x
        mask = dists < cut_off_distance * 5
        if mask.any():
            com_vel_x = np.sum(vels[mask, 0] * masses[mask]) / np.sum(masses[mask])
            com_vel_y = np.sum(vels[mask, 1] * masses[mask]) / np.sum(masses[mask])
            com_vel_z = np.sum(vels[mask, 2] * masses[mask]) / np.sum(masses[mask])
            com_vel = np.array([com_vel_x, com_vel_y, com_vel_z])
            vels_sub_5x[i] -= com_vel

        # 0.5x
        mask = dists < cut_off_distance * 0.5
        if mask.any():
            com_vel_x = np.sum(vels[mask, 0] * masses[mask]) / np.sum(masses[mask])
            com_vel_y = np.sum(vels[mask, 1] * masses[mask]) / np.sum(masses[mask])
            com_vel_z = np.sum(vels[mask, 2] * masses[mask]) / np.sum(masses[mask])
            com_vel = np.array([com_vel_x, com_vel_y, com_vel_z])
            vels_sub_0_5x[i] -= com_vel

        # 0.1x
        mask = dists < cut_off_distance * 0.1
        if mask.any():
            com_vel_x = np.sum(vels[mask, 0] * masses[mask]) / np.sum(masses[mask])
            com_vel_y = np.sum(vels[mask, 1] * masses[mask]) / np.sum(masses[mask])
            com_vel_z = np.sum(vels[mask, 2] * masses[mask]) / np.sum(masses[mask])
            com_vel = np.array([com_vel_x, com_vel_y, com_vel_z])
            vels_sub_0_1x[i] -= com_vel

"""