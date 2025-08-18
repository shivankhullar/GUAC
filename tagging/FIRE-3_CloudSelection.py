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
import os, sys, re, glob

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

def get_center_of_cloud_from_cloud_phinder_data(path_to_hdf5, cloud_num):
    with h5py.File(path_to_hdf5, "r") as f:
        cloud_name = f"Cloud{cloud_num}"
        coords = f[cloud_name]["PartType0"]["Coordinates"]
        part_ids = f[cloud_name]["PartType0"]["ParticleIDs"]
        # Find center of the cloud
        avg_pos = np.mean(coords, axis=0)
        # Find point closest to the center
        distances = np.linalg.norm(coords - avg_pos, axis=1)
        closest_idx = np.argmin(distances)
        center_pt_idx = part_ids[closest_idx]
        print(f"Average pos: {avg_pos}")
        print(f"Closest point has an index of {center_pt_idx}")
        center_pt_pos = coords[closest_idx]
        return center_pt_idx, center_pt_pos


def add_zeros(num, num_digits):
    num_str = str(num)
    while len(num_str) < num_digits:
        num_str = "0" + num_str
    return num_str

def print_arr_info(array):
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    print(f"Array contents: {array}")

def find_snapshot_range(directory):
    # List all files in directory
    all_files = os.listdir(directory)
    
    # Filter files matching snapshot pattern (digits only)
    snapshot_files = []
    pattern = re.compile(r'^snapshot_(\d+)\.hdf5$')
    
    for f in all_files:
        match = pattern.match(f)
        if match:
            snapshot_id = int(match.group(1))  # Convert digits to integer
            snapshot_files.append((snapshot_id, f))
    
    if not snapshot_files:
        return None  # No snapshots found
    
    # Find min/max snapshot IDs
    first_snap = min(snapshot_files, key=lambda x: x[0])
    last_snap = max(snapshot_files, key=lambda x: x[0])
    
    return {
        'first_id': first_snap[0],
        'first_file': first_snap[1],
        'last_id': last_snap[0],
        'last_file': last_snap[1]
    }


#########################################
########### DEFINE PARAMETERS ###########
#########################################

#path = "/mnt/raid-project/murray/khullar/FIRE-3/"
path = sys.argv[1]
sim = sys.argv[2]
output = int(sys.argv[3])
sphere_r = float(sys.argv[4])

if output == 0:
    output = ""
else:
    output = "/output"


path_to_snaps = os.path.join(path, sim, output)
snapshot_num_digits = 3

snaps_range = find_snapshot_range(path_to_snaps)
start_snap = snaps_range['first_id']
last_snap = snaps_range['last_id']

###########################
########### MAIN ##########
###########################

# GET SNAPSHOT NUMBER WITH 1ST STAR FORMING
print(f"Obtaining snapshots from {path_to_snaps}...")
num_digits = snapshot_num_digits
for i in range(start_snap, last_snap+1):
    snap_num = i
    snap_num_str = add_zeros(i, num_digits)
    path_to_snap = path + sim + "/snapshot_" + snap_num_str + ".hdf5"
    with h5py.File(path_to_snap, "r") as f:
        redshift = f["Header"].attrs["Redshift"]
        print(f"Redshift: {redshift}")
        if "PartType4" in f.keys():
            print("Stars found in snapshot {snap_num}")
            break
        elif i == last_snap:
            raise ValueError("No stars found forming")

def get_first_star_pos(coords, star_coords, max_age_idx, sphere_r):
    custom_pos = star_coords[max_age_idx]
    min_norm = min(np.linalg.norm(coords - custom_pos, axis=1))
    print(f"Min norm is: {min_norm}")
    if min_norm > sphere_r:
        #if max_age_idx == sorted_indices[-1]:
        #    raise ValueError("No stars in the region of interest for some reason")
        return 0
    return custom_pos

def get_tracked_parts_ids(coords, custom_pos, part_ids, sphere_r):
    close_gas_coords = coords[np.linalg.norm(coords - custom_pos, axis=1) < sphere_r]
    tracked_close_gas_pIDs = part_ids[np.linalg.norm(coords - custom_pos, axis=1) < sphere_r]
    #close_gas_coords = np.array(close_gas_coords)
    tracked_close_gas_pIDs = np.unique(tracked_close_gas_pIDs, axis=0)
    return tracked_close_gas_pIDs

# GET DATA FROM FIRST-STAR SNAPSHOT
first_star_snap = snap_num
snap_num_str = add_zeros(first_star_snap, num_digits)
path_to_snap = path_to_snaps + "/snapshot_" + snap_num_str + ".hdf5"
loadsnap = lambda array, parttype, snapno : np.array(load_from_snapshot.load_from_snapshot(array, parttype, path_to_snaps, snapno, units_to_physical=True))
loadsnap_first = lambda array, parttype : loadsnap(array, parttype, first_star_snap)
loadsnap_prev = lambda array, parttype : loadsnap(array, parttype, first_star_snap-1)

with h5py.File(path_to_snap, "r") as f:
    coords        = loadsnap_first("Coordinates", 0)
    part_ids      = loadsnap_first("ParticleIDs", 0)
    star_coords   = loadsnap_first("Coordinates", 4)
    star_part_ids = loadsnap_first("ParticleIDs", 4)
    star_ages     = loadsnap_first("StellarFormationTime", 4)
    max_age_idx = np.argmax(star_ages)
    custom_pos = get_first_star_pos(coords, star_coords, max_age_idx, sphere_r)
    first_star_id = -1
    if custom_pos == 0:
        first_star_id = star_part_ids[max_age_idx]
        print(f"No gas particles were found within {sphere_r} of the first star.")
        print(f"Obtaining coordinates from the previous snapshot...")
    else:
        tracked_IDs = get_tracked_parts_ids(coords, custom_pos, part_ids, sphere_r)

if first_star_id != -1:
    with h5py.File(path_to_snap, "r") as f:
        coords         = loadsnap_prev("Coordinates", 0)
        part_ids       = loadsnap_prev("ParticleIDs", 0)
        max_age_idx = np.where(part_ids == first_star_id)[0][0]
        custom_pos = get_first_star_pos(coords, coords, max_age_idx, sphere_r)
        tracked_IDs = get_tracked_parts_ids(coords, custom_pos, part_ids, sphere_r)
        if len(tracked_IDs) <= 1:
            raise ValueError("Something is wrong, no gas particles were found within {sphere_r} of the particle in the snapshot")
        print(f"Found {len(tracked_IDs)} trackable gas particles in the previous snapshot!")

# SAVE IDS TO A BINARY FILE
print("Time to save!")
save_dir = f"{path}{sim}/"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/cloud_tracked_pIDs.npy", tracked_IDs)
print(f"Tracked pIDs saved to {save_dir}/cloud_tracked_pIDs.npy")
with open('snapnum.txt', 'w') as f:
    f.write(str(first_star_snap-1))
