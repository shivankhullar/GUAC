#!/usr/bin/env python

"""
create_cloud_movies: "Create a movie of a cloud in a galaxy."

Usage: create_cloud_movies.py [options]

Options:
    -h, --help                  Show this screen
    --snapdir=<snapdir>         Are snapshots in a snapdir directory? [default: True]
    --threshold=<threshold>     Threshold for cloud selection [default: 0.3]
    --path=<path>               Path to the simulation directory [default: ./]
    --snapnum=<snapnum>         Snapshot number [default: 625]
    --cloud_num=<cloud_num>     Cloud number [default: 40]
    --save_dir=<save_dir>       Directory to save the movie [default: ]
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from meshoid import Meshoid
import matplotlib.pyplot as plt
from docopt import docopt

import yt
import numpy as np
import os
from scipy.interpolate import UnivariateSpline
from multiprocessing import Pool, cpu_count

def get_tracked_cloud_pIDs(chain, params):
    pID_array_list = []
    for i in range(len(chain.cloud_list)):
        cloud_num = chain.cloud_nums[i]
        snapnum = chain.snap_nums[i]
        pID_array, _, _ = get_cloud_quants_hdf5_pIDs(cloud_num, snapnum, params)
        pID_array_list.append(pID_array)
    all_pIDs = np.concatenate(pID_array_list, axis=0)  
    return np.unique(all_pIDs, axis=0)

def get_maximum_cloud_reff(chain, params):
    max_reff = 0
    for i in range(len(chain.cloud_list)):
        snap_num = chain.snap_nums[i]
        cloud_num = chain.cloud_nums[i]
        _, _, cloud_reff, _, _ = get_cloud_quick_info(snap_num, params.nmin, params.vir, cloud_num, params)
        if i == 0:
            init_cloud_reff = cloud_reff
        if cloud_reff > max_reff:
            max_reff = cloud_reff
    cloud_reff_factor = (max_reff / init_cloud_reff) * 1.2
    return max_reff, cloud_reff_factor








def process_single_snapshot(args):
    snap_num, tracked_cloud_pID_array, params, chain = args
    try:
        print (f"Processing snapshot {snap_num}...")
        snap_data = get_snap_data(params, snap_num, gal_quants=True)
        snap_data_pID_array = np.array([snap_data['pIDs'], snap_data['pIDgennum'], snap_data['pIDchilds']]).T

        check = np.isin(snap_data_pID_array, tracked_cloud_pID_array)
        indices = np.where(check.all(axis=1))[0]
        if len(indices) == 0:
            return None
        cloud_coords = snap_data['coords'][indices]
        cloud_masses = snap_data['masses'][indices]
        cloud_dens = snap_data['dens'][indices]

        max_reff, _ = get_maximum_cloud_reff(chain, params)

        median_coords = np.median(cloud_coords, axis=0)
        distances = np.linalg.norm(cloud_coords - median_coords, axis=1)
        new_indices = np.where(distances < 2 * max_reff)[0]
        cloud_coords = cloud_coords[new_indices]

        if cloud_coords.shape[0] == 0:
            return None

        #x_min, x_max, y_min, y_max, z_min, z_max = cloud_coords[:,0].min(), cloud_coords[:,0].max(), \
        #                                                cloud_coords[:,1].min(), cloud_coords[:,1].max(), \
        #                                                cloud_coords[:,2].min(), cloud_coords[:,2].max()
        
        #box_size = max([x_max-x_min, y_max-y_min, z_max-z_min])*2 #*cloud_reff_factor
        #box_size = box_size
        #x_mean = (x_min+x_max)/2
        #y_mean = (y_min+y_max)/2
        #z_mean = (z_min+z_max)/2
        
        x_mean = np.median(cloud_coords[:, 0])
        y_mean = np.median(cloud_coords[:, 1])
        z_mean = np.median(cloud_coords[:, 2])

        #x_mean = cloud_coords[:, 0].mean()
        #y_mean = cloud_coords[:, 1].mean()
        #z_mean = cloud_coords[:, 2].mean()

        return np.array([snap_num, x_mean, y_mean, z_mean])
    
    except Exception as e:
        print(f"Error processing snapshot {snap_num}: {e}")
        return None




def get_cloud_box_centers(params, selected_cloud_nums, selected_cloud_list, selected_snap):
    all_box_centers = []
    for cloud_num in selected_cloud_nums:
        chain = CloudChain(cloud_num, selected_snap, params)
        tracked_cloud_pID_array = get_tracked_cloud_pIDs(chain, params)

        pool_args = [(snap_num, tracked_cloud_pID_array, params, chain)
                     for snap_num in range(params.start_snap, params.last_snap + 1)]

        num_processes = cpu_count() #8               #cpu_count()
        with Pool(processes=num_processes) as pool:
            box_centers = pool.map(process_single_snapshot, pool_args)

        box_centers = [bc for bc in box_centers if bc is not None]
        all_box_centers.append(np.array(box_centers))
    return np.concatenate(all_box_centers, axis=0)





def smooth_and_save_box_centers(params, cloud_name, save_dir, box_centers, smoothing_factor=4e4, sorted=False):
    if not sorted:
        box_centers = box_centers[np.argsort(box_centers[:, 0])]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f'{save_dir}/{cloud_name}_box_centers.npy', box_centers)

    snaps = box_centers[:, 0]
    x = box_centers[:, 1]
    y = box_centers[:, 2]
    z = box_centers[:, 3]

    x_spline = UnivariateSpline(snaps, x, k=3, s=smoothing_factor)
    y_spline = UnivariateSpline(snaps, y, k=3, s=smoothing_factor)
    z_spline = UnivariateSpline(snaps, z, k=3, s=smoothing_factor)

    x_smooth = x_spline(snaps)
    y_smooth = y_spline(snaps)
    z_smooth = z_spline(snaps)

    smoothed_box_centers = np.column_stack((snaps, x_smooth, y_smooth, z_smooth))
    np.save(f'{save_dir}/{cloud_name}_box_centers_smoothed.npy', smoothed_box_centers)




def load_box_center(params, cloud_name, save_dir, snap=None):
    box_centers = np.load(f'{save_dir}/{cloud_name}_box_centers.npy')
    smoothed_box_centers = np.load(f'{save_dir}/{cloud_name}_box_centers_smoothed.npy')
    if snap is not None:
        snap_ind = np.where(box_centers[:, 0] == snap)[0][0]
        return box_centers[snap_ind], smoothed_box_centers[snap_ind]
    return box_centers, smoothed_box_centers








if __name__ == "__main__":
    args = docopt(__doc__)
    snapdir = args['--snapdir']
    threshold = args['--threshold']
    path = args['--path']
    snapnum = int(args['--snapnum'])
    cloud_num = int(args['--cloud_num'])
    save_dir = args['--save_dir']

    path = "/mnt/raid-project/murray/khullar/FIRE-3/"
    start_snap = 500
    last_snap = 900
    linked_filename_prefix = "Linked_Clouds_"
    cloud_num_digits = 4
    snapshot_num_digits = 4
    cloud_prefix = "Cloud"
    hdf5_file_prefix = 'Clouds_'
    sim = 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5'
    age_cut = 5
    dat_file_header_size = 11
    snapshot_prefix = "Snap"
    star_data_sub_dir = "StarData/"
    gas_data_sub_dir = "GasData/"
    cph_sub_dir = "CloudPhinderData/"
    r_gal = 25
    h = 0.4
    nmin = 1
    vir = 10
    frac_thresh = 'thresh' + threshold
    sub_dir = f"CloudPhinderData/n{nmin}_alpha{vir}/"
    vels_sub_dir = 'vel_sub/'
    gal_quants_sub_dir = 'gal_quants/'

    params = Params(path=path, sub_dir=sub_dir, start_snap=start_snap, last_snap=last_snap,
                    filename_prefix=linked_filename_prefix, cloud_prefix=cloud_prefix,
                    hdf5_file_prefix=hdf5_file_prefix, frac_thresh=frac_thresh, sim=sim,
                    r_gal=r_gal, h=h, gal_quants_sub_dir=gal_quants_sub_dir, vels_sub_dir=vels_sub_dir,
                    phinder_sub_dir=cph_sub_dir, age_cut=age_cut,
                    dat_file_header_size=dat_file_header_size, nmin=nmin, vir=vir,
                    cloud_num_digits=cloud_num_digits, snapshot_num_digits=snapshot_num_digits, verbose=False)

    params.linked_filename = f"{params.path}{params.sub_dir}{params.filename_prefix}n{params.nmin}_alpha{params.vir}_{params.frac_thresh}_{params.start_snap}_{params.last_snap}_names.txt"

    selected_snap = snapnum
    selected_cloud_nums = np.array([cloud_num])
    selected_cloud_list = [f'Cloud{cloud_num:04d}Snap{selected_snap}']

    box_centers = get_cloud_box_centers(params, selected_cloud_nums, selected_cloud_list, selected_snap)

    if save_dir == '':
        save_dir = f"{params.path}cloud_movies/n{params.nmin}_alpha{params.vir}/box_center_data/"

    print("Saving box centers to:", save_dir)
    smooth_and_save_box_centers(params, selected_cloud_list[0], save_dir, box_centers, smoothing_factor=4e4, sorted=False)
