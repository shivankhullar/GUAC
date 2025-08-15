import numpy as np
import h5py
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
                     params, box_size, cloud_name, snap_num, x_mean, y_mean, z_mean, max_reff, chain, save_dir, custom_pos=None):
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
    # Set ticks outwards and big
    ax.tick_params(axis='both', which='major', direction='out', length=10, width=2, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='out', length=5, width=1, labelsize=12)
    ax.set_xlabel('X (kpc)', fontsize=18)
    ax.set_ylabel('Y (kpc)', fontsize=18)
    #ax.axis('off')

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

    age_cut = 10 #5
    young = final_star_ages < age_cut
    ax.scatter(final_star_coords[young, 0], final_star_coords[young, 1], s=50, c=final_star_ages[young], cmap='bwr', marker='*', alpha=1)

    if custom_pos is not None:
        ax.scatter(custom_pos[0], custom_pos[1], s=100, c='g', marker='X')#, label='Custom Position')
        #ax.legend(fontsize=14)


    fontprops = FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData, max_reff, f'{max_reff:.2f} kpc', 'upper left',
                                pad=1, color='white', frameon=False, size_vertical=0.005,
                                fontproperties=fontprops)
    ax.add_artist(scalebar)

    #movie_save_dir = save_dir + f"movie_data/{cloud_name}/"
    #os.makedirs(movie_save_dir, exist_ok=True)

    plt.tight_layout()
    #plt.savefig(f"{movie_save_dir}/snapshot_{snap_num}.png", dpi=300)
    plt.show()
    plt.close()

#path = "/mnt/raid-project/murray/khullar/FIRE-3/"
path = "/fs/lustre/scratch/vpustovoit/SHIVAN/CCA_DATA/"
start_snap = 6
last_snap = 6
linked_filename_prefix = "Linked_Clouds_"
cloud_num_digits = 4
snapshot_num_digits = 3
cloud_prefix = "Cloud"
hdf5_file_prefix = 'Clouds_'
#sim = 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5'
sim = 'm12f'
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

threshold = '0.5'
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


cloud_nums = np.array([0, 1])

def get_center_of_cloud_from_cloud_phinder_data(path_to_hdf5, cloud_num):
    with h5py.File(path_to_hdf5, "r") as f:
        cloud_name = f"Cloud{cloud_num}"
        coords = f[cloud_name]["PartType0"]["Coordinates"]
        part_ids = f[cloud_name]["PartType0"]["ParticleIDs"]
        avg_pos = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - avg_pos, axis=1)
        closest_idx = np.argmin(distances)
        center_pt_idx = part_ids[closest_idx]
        print(f"Average pos: {avg_pos}")
        print(f"Closest point has an index of {center_pt_idx}")


path_to_hdf5 = path + sim + "/Clouds_6_n0.05_alpha5.hdf5"
for i in range(0, len(cloud_nums)):
    cloud_num = cloud_nums[i]
    get_center_of_cloud_from_cloud_phinder_data(path_to_hdf5, cloud_num)
    """
    selected_snap=6
    cloud_name = f'Cloud{cloud_num:04d}Snap{selected_snap}'

    chain = CloudChain(cloud_num, selected_snap, params)
    cnum = chain.cloud_nums[chain.snap_nums.index(selected_snap)]
    _, _, coords, _, _ = get_cloud_quants_hdf5(cnum, selected_snap, params, no_temps=True, projection=True)

    median_gas_pos = np.median(coords, axis=0)
    #print(f"Median gas position: {median_gas_pos}")

    gal_center = get_galaxy_centre(params, selected_snap)
    proj_matrix = get_galaxy_proj_matrix(params, selected_snap)

    gal_center_proj = np.matmul(proj_matrix, gal_center)
    print(f"Galaxy center projected coordinates: {gal_center_proj}")

    gal_dist = np.linalg.norm(gal_center_proj - median_gas_pos)
    print(f"Distance from galaxy center to {cloud_name}: {gal_dist:.2f} kpc")

    """
