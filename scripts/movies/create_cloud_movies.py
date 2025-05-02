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

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import multiprocessing as mp
from docopt import docopt

from meshoid import Meshoid
from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *


import yt
from yt.units import parsec, Msun
from yt.utilities.cosmology import Cosmology


from matplotlib.colors import LogNorm
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os

import colorcet as cc


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
    return max_reff, (4 * max_reff)

def load_box_center(params, cloud_name, save_dir, snap=None):
    box_centers = np.load(f'{save_dir}/{cloud_name}_box_centers.npy')
    smoothed_box_centers = np.load(f'{save_dir}/{cloud_name}_box_centers_smoothed.npy')
    if snap is not None:
        snap_ind = np.where(box_centers[:, 0] == snap)[0][0]
        return box_centers[snap_ind], smoothed_box_centers[snap_ind]
    return box_centers, smoothed_box_centers

def make_cloud_plot(final_gas_coords, final_gas_masses, final_gas_hsmls, final_star_coords, final_star_masses, final_star_ages,
                     params, box_size, cloud_name, snap_num, x_mean, y_mean, z_mean, max_reff, chain, save_dir):
    cb_vmin = 1
    cb_vmax = 5e3

    fig, ax = plt.subplots(figsize=(8, 8))
    res = 1024
    center = np.array([x_mean, y_mean, z_mean])
    min_pos = center - box_size / 2
    max_pos = center + box_size / 2
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    M = Meshoid(final_gas_coords, final_gas_masses, final_gas_hsmls)
    sigma_gas = M.SurfaceDensity(M.m, center=center, size=box_size, res=res) * 1e4

    ax.pcolormesh(X, Y, sigma_gas, norm=colors.LogNorm(vmin=cb_vmin, vmax=cb_vmax), cmap='cet_fire')
    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    ax.axis('off')

    ax.scatter(final_gas_coords[:, 0], final_gas_coords[:, 1], s=0.1, c='w', alpha=0.5)

    if snap_num in chain.snap_nums:
        cnum = chain.cloud_nums[chain.snap_nums.index(snap_num)]
        _, _, coords, _, _ = get_cloud_quants_hdf5(cnum, snap_num, params, no_temps=True, projection=True)
        #_, _, coords, _, _, _, _ = get_cloud_quants_hdf5(cnum, snap_num, params, no_temps=True)
        ax.scatter(coords[:, 0], coords[:, 1], s=0.2, c='k', alpha=0.5)
        hull = ConvexHull(coords[:, :2])
        bound_x = np.append(coords[hull.vertices, 0], coords[hull.vertices[0], 0])
        bound_y = np.append(coords[hull.vertices, 1], coords[hull.vertices[0], 1])
        ax.plot(bound_x, bound_y, color='w', lw=1, linestyle='--')

    age_cut = 5
    young = final_star_ages < age_cut
    ax.scatter(final_star_coords[young, 0], final_star_coords[young, 1], s=50, c=final_star_ages[young], cmap='bwr', marker='*', alpha=1)

    fontprops = FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData, max_reff, f'{max_reff:.2f} kpc', 'upper left',
                                pad=1, color='white', frameon=False, size_vertical=0.005,
                                fontproperties=fontprops)
    ax.add_artist(scalebar)
    movie_save_dir = save_dir + f"movie_data/{cloud_name}/"
    #save_dir = f"{params.path}{params.sub_dir}selected_cloud_movies/{chain.search_key}/"
    os.makedirs(movie_save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{movie_save_dir}/snapshot_{snap_num}.png", dpi=300)
    plt.close()

def process_snapshot_movie(args):
    snap_num, params, cloud_name, cloud_name, selected_snap, save_dir = args
    try:
        snap_data = get_snap_data(params, snap_num, gal_quants=True)

        star_coords = snap_data['star_coords']
        star_masses = snap_data['star_masses']
        star_ages = snap_data['star_ages']
        coords = snap_data['coords']
        vels = snap_data['vels']
        masses = snap_data['masses']
        dens = snap_data['dens']
        hsmls = snap_data['hsml']
        snap_data_pID_array = np.array([snap_data['pIDs'], snap_data['pIDgennum'], snap_data['pIDchilds']]).T

        #for cloud_count, cloud_num in enumerate(selected_cloud_nums):
        chain = CloudChain(cloud_num, selected_snap, params)
        tracked_cloud_pID_array = get_tracked_cloud_pIDs(chain, params)

        check = np.isin(snap_data_pID_array, tracked_cloud_pID_array)
        indices = np.where(check.all(axis=1))[0]
        if len(indices) == 0:
            print (f"Snapshot {snap_num} has no valid cloud pIDs.")
            #return
        #    continue

        cloud_coords = coords[indices]
        cloud_vels = vels[indices]
        cloud_masses = masses[indices]
        cloud_dens = dens[indices]
        cloud_hsmls = hsmls[indices]

        max_reff, box_size = get_maximum_cloud_reff(chain, params)

        median_coords = np.median(cloud_coords, axis=0)
        distances = np.linalg.norm(cloud_coords - median_coords, axis=1)
        new_indices = np.where(distances < 2 * max_reff)[0]
        if len(new_indices) == 0:
            print (f"Snapshot {snap_num} has no valid cloud coordinates.")
        #    continue

        cloud_coords = cloud_coords[new_indices]
        cloud_vels = cloud_vels[new_indices]
        cloud_masses = cloud_masses[new_indices]
        cloud_dens = cloud_dens[new_indices]
        cloud_hsmls = cloud_hsmls[new_indices]

        box_save_dir = save_dir+"box_center_data/"
        center, smoothed_center = load_box_center(params, cloud_name, box_save_dir, snap=snap_num)
        #x_mean, y_mean, z_mean = smoothed_center[1:]
        x_mean, y_mean, z_mean = center[1:]
        print (f"Snapshot {snap_num} has box center: {x_mean}, {y_mean}, {z_mean}")

        # Now we will get all the gas particles and star particles within this box from the snapshot data.
        inds_x = np.where((coords[:,0]>x_mean-box_size/2)&(coords[:,0]<x_mean+box_size/2))[0]
        inds_y = np.where((coords[:,1]>y_mean-box_size/2)&(coords[:,1]<y_mean+box_size/2))[0]
        inds_z = np.where((coords[:,2]>z_mean-box_size/2)&(coords[:,2]<z_mean+box_size/2))[0]
        final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)
        if len(final_inds) == 0:
            print (f"Snapshot {snap_num} has no valid coordinates. Weird.")
            #continue
        coords_x = np.take(coords[:,0], final_inds)
        coords_y = np.take(coords[:,1], final_inds)
        coords_z = np.take(coords[:,2], final_inds)
        final_gas_coords = np.array([coords_x, coords_y, coords_z]).T
        final_gas_masses = np.take(masses, final_inds)
        final_gas_hsmls = np.take(hsmls, final_inds)

        # Get the star particles
        inds_x = np.where((star_coords[:,0]>x_mean-box_size/2)&(star_coords[:,0]<x_mean+box_size/2))[0]
        inds_y = np.where((star_coords[:,1]>y_mean-box_size/2)&(star_coords[:,1]<y_mean+box_size/2))[0]
        inds_z = np.where((star_coords[:,2]>z_mean-box_size/2)&(star_coords[:,2]<z_mean+box_size/2))[0]
        final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)

        if len(final_inds) == 0:
            print (f"Snapshot {snap_num} has no valid star coordinates. Weird.")
        
        coords_x = np.take(star_coords[:,0], final_inds)
        coords_y = np.take(star_coords[:,1], final_inds)
        coords_z = np.take(star_coords[:,2], final_inds)
        final_star_coords = np.array([coords_x, coords_y, coords_z]).T
        final_star_masses = np.take(star_masses, final_inds)
        final_star_ages = np.take(star_ages, final_inds)

        
        make_cloud_plot(final_gas_coords, final_gas_masses, final_gas_hsmls, final_star_coords, final_star_masses,
                        final_star_ages, params, box_size, cloud_name, snap_num,
                        x_mean, y_mean, z_mean, max_reff, chain, save_dir)
    except Exception as e:
        print(f"Snapshot {snap_num} failed: {e}")


def make_movies_of_clouds(params, selected_cloud_nums, cloud_name, selected_snap, save_dir):
    snap_args = [
        (snap_num, params, selected_cloud_nums, cloud_name, selected_snap, save_dir)
        for snap_num in range(params.start_snap, params.last_snap + 1)
    ]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_snapshot_movie, snap_args)

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
                    cloud_num_digits=cloud_num_digits, snapshot_num_digits=snapshot_num_digits)

    params.linked_filename = f"{params.path}{params.sub_dir}{params.filename_prefix}n{params.nmin}_alpha{params.vir}_{params.frac_thresh}_{params.start_snap}_{params.last_snap}_names.txt"

    selected_snap = snapnum
    selected_cloud_nums = np.array([cloud_num])
    #selected_cloud_list = [f'Cloud{cloud_num:04d}Snap{snapnum}']
    cloud_name = f'Cloud{cloud_num:04d}Snap{selected_snap}'

    if save_dir == '':
        save_dir = f"{params.path}cloud_movies/n{params.nmin}_alpha{params.vir}/" #/movie_data/"

    make_movies_of_clouds(params, selected_cloud_nums, cloud_name, selected_snap, save_dir)
