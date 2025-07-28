"""
This file contains utility functions to get Supernovae related quantities.

Author: Shivan Khullar
Date: July 2025
"""

from generic_utils.fire_utils import *
import generic_utils.constants as const
from void_utils.calculate_void_quants import *
from void_utils.io_utils import *


import h5py
import yt
from yt.units import parsec, Msun
import glob




def read_scale_factors_file(params):
    file = params.path+"snapshot_scale_factors.txt"
    scale_factors = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                scale_factors.append(float(line.strip()))
    return scale_factors[1:]
    
def get_current_scale_factor(params, snapnum):
    scale_factors = read_scale_factors_file(params)
    return scale_factors[snapnum-params.start_snap]    




def read_SNe_info(filename):
    SNe = []
    with open(filename,"r") as f:
        for line in f:
            if line.startswith("SNe: "):
                SNe.append(line[5:].strip())
                #print(line[5:].strip())
    return SNe


def get_SNe_data_from_file(params):
    #, gal_quants4, snap_interval=10):
    # Read the SNe data from the file
    filename = params.path+"shell_30-01-2025.out"
    SNe_info = read_SNe_info(filename)

    pIDs = []
    pos_x = []
    pos_y = []
    pos_z = []
    vel_x = []
    vel_y = []
    vel_z = []
    mass = []
    nums = []
    tasks = []
    times = []


    for i in range(0, len(SNe_info)):
        SNe_info[i] = SNe_info[i].split(', ')
        pIDs.append(int(SNe_info[i][0]))
        pos_x.append(float(SNe_info[i][1]))
        pos_y.append(float(SNe_info[i][2]))
        pos_z.append(float(SNe_info[i][3]))
        vel_x.append(float(SNe_info[i][4]))
        vel_y.append(float(SNe_info[i][5]))
        vel_z.append(float(SNe_info[i][6]))
        mass.append(float(SNe_info[i][7]))
        nums.append(int(SNe_info[i][8]))
        tasks.append(int(SNe_info[i][9]))
        times.append(float(SNe_info[i][10]))

    pIDs = np.array(pIDs)
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    pos_z = np.array(pos_z)
    vel_x = np.array(vel_x)
    vel_y = np.array(vel_y)
    vel_z = np.array(vel_z)
    mass = np.array(mass)
    nums = np.array(nums)
    tasks = np.array(tasks)
    times = np.array(times)


    snapshot_times = np.array(read_scale_factors_file(params))

    return pIDs, times, snapshot_times





def get_SNe_data_for_snapshot(params, snapnum, pIDs, times, snapshot_times, gal_quants4, snap_interval=10):
    lower_index = int(snapnum-params.start_snap-snap_interval/2) if snapnum-params.start_snap-snap_interval/2 >= 0 else 0
    upper_index = int(snapnum-params.start_snap+snap_interval/2)
    if snap_interval == 1:
        upper_index = lower_index + 1
    if upper_index >= len(snapshot_times):
        upper_index = len(snapshot_times) - 1

    mask = (times > snapshot_times[lower_index]) & (times <= snapshot_times[upper_index])
    filtered_pIDs = pIDs[mask]
    filtered_times = times[mask]

    unique_pIDs, SNe_counts = np.unique(filtered_pIDs, return_counts=True)
    SNe_star_pIDs, inds1, inds2 = np.intersect1d(gal_quants4.data["ParticleIDs"], unique_pIDs, return_indices=True)

    SNe_star_pos_x = gal_quants4.data["Coordinates"][:,0][inds1]
    SNe_star_pos_y = gal_quants4.data["Coordinates"][:,1][inds1]
    SNe_star_pos_z = gal_quants4.data["Coordinates"][:,2][inds1]
    SNe_star_counts = SNe_counts[inds2]

    snap_idx = snapnum - params.start_snap
    time_to_snap_idx = np.searchsorted(snapshot_times, filtered_times)
    max_diff = snap_interval / 2
    importance_scores = []
    for pid, N in zip(SNe_star_pIDs, SNe_star_counts):
        pid_mask = (filtered_pIDs == pid)
        snap_diffs = np.abs(time_to_snap_idx[pid_mask] - snap_idx)
        if len(snap_diffs) == 0:
            score = 1 + N
        else:
            min_diff = np.min(snap_diffs)
            interp = (10 - 1) * (1 - min(min_diff, max_diff) / max_diff) + 1
            score = interp + N
        importance_scores.append(score)
    importance_scores = np.array(importance_scores)

    return SNe_star_pos_x, SNe_star_pos_y, SNe_star_pos_z, SNe_star_counts, importance_scores