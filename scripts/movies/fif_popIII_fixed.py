#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']      = '1'

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from hybrid_sims_utils.read_snap import *
from hybrid_sims_utils.hybrid_utils import *
from constants import *

from meshoid import Meshoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import colorcet as cc
import cmasher as cmr
from meshoid import GridSurfaceDensity
import h5py

import glob
import re
import os
import argparse
from multiprocessing import Pool
import time
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d


# ===========================================================================
# CONFIG
# ===========================================================================
path_full  = "/mnt/ceph/users/skhullar/projects/SFIRE/m12f/"
sim_cutout = "output_cutout"

"""
save_path = "/mnt/home/skhullar/projects/SFIRE/m12f/movies/popIII_fixed/"

SNAP_START = 28
SNAP_END   = 611

PC_TO_KPC      = 1e-3
FIXED_BOX      = 0.2 * PC_TO_KPC   # 0.2 pc in kpc
SCALEBAR       = 0.1 * PC_TO_KPC   # 0.1 pc in kpc
RUNAWAY_KPC    = 1.0 * PC_TO_KPC   # ignore stars > 1 pc from cluster COM
"""

save_path = "/mnt/home/skhullar/projects/SFIRE/m12f/movies/popIII_fixed_bigbox/"

SNAP_START = 28
SNAP_END   = 1067

PC_TO_KPC      = 1e-3
FIXED_BOX      = 0.4 * PC_TO_KPC   # 0.4 pc in kpc
SCALEBAR       = 0.1 * PC_TO_KPC   # 0.1 pc in kpc
RUNAWAY_KPC    = 0.5 * PC_TO_KPC   # ignore stars > 0.5 pc from cluster COM

GAS_FIELDS = ['Masses', 'Coordinates', 'SmoothingLength', 'Velocities',
              'Density', 'InternalEnergy', 'ElectronAbundance', 'Temperature']
# ===========================================================================


# ===========================================================================
# Frame sequence — one frame per snapshot, fixed box
# ===========================================================================

total_frames        = SNAP_END - SNAP_START + 1   # 584
snap_nums_per_frame = np.arange(SNAP_START, SNAP_END + 1, dtype=int)


# ===========================================================================
# Center finder
# ===========================================================================

def find_center(pdata, stardata):
    """Center finder anchored to the gas density peak.

    1. Find the densest gas particle (guaranteed correct units).
    2. If sink particles exist within RUNAWAY_KPC of that gas peak, use their
       mass-weighted COM — these are the bound cluster sinks.
    3. Otherwise fall back to the gas density peak.

    Anchoring to the gas peak means a star coordinate unit mismatch (comoving
    vs physical) is automatically detected: all sinks would appear > 1 pc away
    and the fallback kicks in."""
    gas_peak = pdata['Coordinates'][np.argmax(pdata['Density'])].copy()

    if stardata is None or len(stardata.get('Masses', [])) == 0:
        return gas_peak

    coords = stardata['Coordinates']
    masses = stardata['Masses']

    # Accept only sinks within RUNAWAY_KPC of the gas density peak
    near = np.linalg.norm(coords - gas_peak, axis=1) < RUNAWAY_KPC
    if near.sum() == 0:
        return gas_peak  # no sinks near gas peak — unit mismatch or all runaways

    return np.sum(coords[near] * masses[near, None], axis=0) / np.sum(masses[near])


# ===========================================================================
# Per-frame data loading — returns raw (uncentered) coordinates + raw COM
# ===========================================================================

def get_frame_data(snap_num):
    hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
        sim_cutout, path_full, snap_num,
        movie_tag=True, custom_gas_fields=GAS_FIELDS,
        snapshot_suffix='', snapdir=False, refinement_tag=False, full_tag=False)
    hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)

    a        = hdr["Time"]
    redshift = 1.0 / a - 1.0
    time_Myr = convert_scale_factor_to_time(a, hdr)

    raw_com = find_center(pdata, stardata)

    return pdata, stardata, fsd, raw_com, time_Myr, redshift


# ===========================================================================
# Collect colorbar ranges
# ===========================================================================

def collect_colorbar_ranges(count, com, pdata, gas_dists, res=1920):
    dist_cut_off = FIXED_BOX * 2

    mask   = gas_dists < dist_cut_off
    pos    = pdata['Coordinates'][mask]
    masses = pdata['Masses'][mask]
    hsml   = pdata['SmoothingLength'][mask]

    sigma_gas = GridSurfaceDensity(masses, pos, hsml, com, FIXED_BOX, res=res)

    if np.any(np.isnan(sigma_gas)):
        print(f"Frame {count:04d}: NaN in sigma_gas — skipping.")
        return None

    sigma_valid = sigma_gas[sigma_gas > 0]
    if len(sigma_valid) == 0:
        print(f"Frame {count:04d}: empty sigma_gas — skipping.")
        return None

    ranges = {
        'sigma_gas_min': float(sigma_valid.min()),
        'sigma_gas_max': float(sigma_valid.max()),
    }
    print(f"Frame {count:04d}: snap={snap_nums_per_frame[count]} | "
          f"sigma=[{ranges['sigma_gas_min']:.2e}, {ranges['sigma_gas_max']:.2e}]")
    return ranges


# ===========================================================================
# Save / load / interpolate colorbar ranges
# ===========================================================================

def save_colorbar_ranges(ranges_dict, save_path):
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')
    with open(ranges_file, 'w') as f:
        f.write("# Frame sigma_gas_min sigma_gas_max\n")
        for frame in sorted(ranges_dict.keys()):
            r = ranges_dict[frame]
            f.write(f"{frame} "
                    f"{r['sigma_gas_min']:.10e} {r['sigma_gas_max']:.10e}\n")
    print(f"Saved colorbar ranges to {ranges_file}")


def load_and_interpolate_colorbar_ranges(save_path, num_frames):
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')
    if not os.path.exists(ranges_file):
        print(f"Error: {ranges_file} not found. Run in 'collect' mode first.")
        return None

    data          = np.loadtxt(ranges_file)
    frames        = data[:, 0].astype(int)
    sigma_gas_min = data[:, 1]
    sigma_gas_max = data[:, 2]

    sf = len(frames) * 0.5

    interp_sigma_min = UnivariateSpline(frames, np.log10(sigma_gas_min), k=3, s=sf)
    interp_sigma_max = UnivariateSpline(frames, np.log10(sigma_gas_max), k=3, s=sf)

    interpolated_ranges = {}
    for frame in np.arange(num_frames):
        interpolated_ranges[frame] = {
            'sigma_gas_min': 10 ** float(interp_sigma_min(frame)),
            'sigma_gas_max': 10 ** float(interp_sigma_max(frame)),
        }
    print(f"Loaded and smoothed colorbar ranges for {num_frames} frames")
    return interpolated_ranges


# ===========================================================================
# Save / load / smooth COM positions
# ===========================================================================

def save_com_positions(com_dict, save_path):
    com_file = os.path.join(save_path, 'com_positions.txt')
    with open(com_file, 'w') as f:
        f.write("# Frame x y z\n")
        for frame in sorted(com_dict.keys()):
            x, y, z = com_dict[frame]
            f.write(f"{frame} {x:.10e} {y:.10e} {z:.10e}\n")
    print(f"Saved raw COM positions to {com_file}")


def load_and_smooth_coms(save_path, num_frames, kernel_size=15):
    com_file = os.path.join(save_path, 'com_positions.txt')
    if not os.path.exists(com_file):
        return None

    data   = np.loadtxt(com_file)
    frames = data[:, 0].astype(int)
    raw    = data[:, 1:]   # shape (N, 3)

    # Pre-star frames have raw COM = (0,0,0); star frames have COM in physical kpc.
    # Smoothing across this jump contaminates the transition frames — smooth only
    # the star phase separately.
    zero_mask = np.all(np.abs(raw) < 1e-30, axis=1)
    smoothed  = raw.copy()
    star_idxs = np.where(~zero_mask)[0]
    if len(star_idxs) > 0:
        first_star = star_idxs[0]
        for i in range(3):
            smoothed[first_star:, i] = uniform_filter1d(
                raw[first_star:, i], size=kernel_size, mode='nearest')
        print(f"Smoothed {len(star_idxs)} star-phase COMs (kernel={kernel_size}); "
              f"{first_star} pre-star frames kept at raw (0,0,0)")
    else:
        print("No star-phase COMs found — all frames use raw COM")

    # Build frame → smoothed COM lookup, filling gaps by nearest neighbour
    com_map   = {int(frames[fi]): smoothed[fi] for fi in range(len(frames))}
    com_array = np.zeros((num_frames, 3))
    for f in range(num_frames):
        if f in com_map:
            com_array[f] = com_map[f]
        else:
            nearest = min(com_map.keys(), key=lambda k: abs(k - f))
            com_array[f] = com_map[nearest]

    return com_array


# ===========================================================================
# Render frame
# ===========================================================================

def plot_image(count, com, pdata, gas_dists, stardata, fire_stardata,
               time_Myr, redshift, res=1920, save_path='./',
               colorbar_limits=None):

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    dist_cut_off = FIXED_BOX * 2

    mask   = gas_dists < dist_cut_off
    pos    = pdata['Coordinates'][mask]
    masses = pdata['Masses'][mask]
    hsml   = pdata['SmoothingLength'][mask]

    sigma_gas = GridSurfaceDensity(masses, pos, hsml, com, FIXED_BOX, res=res)

    if colorbar_limits is not None:
        sigma_vmin = colorbar_limits['sigma_gas_min']
        sigma_vmax = colorbar_limits['sigma_gas_max']
    else:
        s_v = sigma_gas[sigma_gas > 0]
        sigma_vmin = float(s_v.min()) if len(s_v) > 0 else 1e-10
        sigma_vmax = float(s_v.max()) if len(s_v) > 0 else 1e10

    sigma_vmin = max(float(sigma_vmin), 1e-30)
    sigma_vmax = max(float(sigma_vmax), sigma_vmin * 10)

    ax.imshow(sigma_gas.T, origin='lower',
              norm=LogNorm(vmin=sigma_vmin, vmax=sigma_vmax),
              cmap=cmr.eclipse, alpha=1.0)

    # Star overlay
    for sdata in [stardata, fire_stardata]:
        if sdata is None or len(sdata.get('Coordinates', [])) == 0:
            continue
        sc      = sdata['Coordinates']
        smasses = sdata.get('Masses', np.ones(len(sc)) * 1e-10)
        in_view = ((np.abs(sc[:, 0]) < FIXED_BOX / 2) &
                   (np.abs(sc[:, 1]) < FIXED_BOX / 2))
        if in_view.sum() == 0:
            continue
        pix_x  = (sc[in_view, 0] + FIXED_BOX / 2) / FIXED_BOX * res
        pix_y  = (sc[in_view, 1] + FIXED_BOX / 2) / FIXED_BOX * res
        s_vals = np.maximum(60.0 * np.log10(np.maximum(smasses[in_view] * 1e10, 1.0)), 60.0)
        ax.scatter(pix_x, pix_y, s=s_vals, c='yellow', marker='*',
                   zorder=5, edgecolors='black', linewidths=0.5)

    plt.xticks([])
    plt.yticks([])

    # Scale bar — fixed 0.1 pc
    scale_bar_pix = SCALEBAR * res / FIXED_BOX
    fontprops     = fm.FontProperties(size=18)
    scalebar      = AnchoredSizeBar(ax.transData,
                                    scale_bar_pix, '0.1 pc', 'upper left',
                                    pad=1, color='white', frameon=False,
                                    size_vertical=scale_bar_pix / 100,
                                    fontproperties=fontprops)
    ax.add_artist(scalebar)

    # Time + redshift annotation
    dt_kyr = max(0.0, (time_Myr - g_ref_time_Myr) * 1000.0)
    ax.text(0.98, 0.02,
            f"Time = {dt_kyr:.2f} kyr    z = {redshift:.2f}",
            transform=ax.transAxes,
            color='white', fontsize=14, ha='right', va='bottom',
            fontproperties=fm.FontProperties(size=14))

    fig.set_facecolor('black')
    plt.tight_layout()

    image_save_path = save_path + 'surf_dens/'
    os.makedirs(image_save_path, exist_ok=True)
    plt.savefig(image_save_path + f"snap_{count:04d}.png", dpi=250)
    plt.close(fig)
    print(f"Saved snap_{count:04d}.png")


# ===========================================================================
# Load snap_028 to get reference time
# ===========================================================================

print("Loading snap_028 to get reference time...")
_hdr28, _pdata28, _sdata28, _fsd28, _, _ = get_snap_data_hybrid(
    sim_cutout, path_full, SNAP_START,
    movie_tag=True, custom_gas_fields=GAS_FIELDS,
    snapshot_suffix='', snapdir=False, refinement_tag=False, full_tag=False)
_hdr28, _pdata28, _sdata28, _fsd28 = convert_units_to_physical(
    _hdr28, _pdata28, _sdata28, _fsd28)

_a28           = _hdr28["Time"]
g_ref_time_Myr = convert_scale_factor_to_time(_a28, _hdr28)
print(f"Reference time (snap_028): {g_ref_time_Myr:.6f} Myr")
del _pdata28, _sdata28, _fsd28


# ===========================================================================
# Mode selection & parallelization
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['collect', 'plot'], default='plot')
_args = parser.parse_args()
MODE = _args.mode

parallelize = True
num_cores   = 128

start_time          = time.time()
completed_count     = 0
colorbar_ranges     = {}
raw_com_dict        = {}
interpolated_ranges = None
smoothed_coms       = None


def collect_wrapper(args):
    global completed_count, start_time
    count, snap_num = args

    pdata_, sdata_, fsd_, raw_com_, _, _ = get_frame_data(snap_num)

    # Center on raw COM for colorbar collection
    pdata_['Coordinates'] -= raw_com_
    gas_dists_ = np.linalg.norm(pdata_['Coordinates'], axis=1)

    ranges = collect_colorbar_ranges(count, np.zeros(3), pdata_, gas_dists_, res=1920)
    completed_count += 1
    if completed_count % 50 == 0:
        elapsed = time.time() - start_time
        avg_t   = elapsed / completed_count
        remain  = total_frames - completed_count
        print(f"Collect {completed_count}/{total_frames} | "
              f"{elapsed/60:.1f} min | ~{avg_t*remain/60:.1f} min left")
    return count, raw_com_, ranges


def plot_wrapper(args):
    global completed_count, start_time, interpolated_ranges, smoothed_coms
    count, snap_num = args

    # Skip frames that already rendered successfully
    image_save_path = save_path + 'surf_dens/'
    out_file = os.path.join(image_save_path, f"snap_{count:04d}.png")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        completed_count += 1
        return

    pdata_, sdata_, fsd_, raw_com_, time_Myr_, redshift_ = get_frame_data(snap_num)

    # Use smoothed COM if available, else fall back to per-frame raw COM
    com_ = smoothed_coms[count] if smoothed_coms is not None else raw_com_
    pdata_['Coordinates'] -= com_
    if sdata_:
        sdata_['Coordinates'] -= com_
    if fsd_:
        fsd_['Coordinates'] -= com_
    gas_dists_ = np.linalg.norm(pdata_['Coordinates'], axis=1)

    colorbar_limits = interpolated_ranges[count] if interpolated_ranges is not None else None
    plot_image(count, np.zeros(3), pdata_, gas_dists_, sdata_, fsd_,
               time_Myr_, redshift_,
               res=1920, save_path=save_path, colorbar_limits=colorbar_limits)

    completed_count += 1
    if completed_count % 50 == 0:
        elapsed = time.time() - start_time
        avg_t   = elapsed / completed_count
        remain  = total_frames - completed_count
        print(f"Plot {completed_count}/{total_frames} | "
              f"{elapsed/60:.1f} min | ~{avg_t*remain/60:.1f} min left")


# ===========================================================================
# Main
# ===========================================================================
os.makedirs(save_path, exist_ok=True)

args_list = [(i, int(snap_nums_per_frame[i])) for i in range(total_frames)]

if MODE == 'collect':
    print("=" * 70)
    print(f"MODE: COLLECT — {total_frames} frames, {num_cores} cores")
    print(f"  box = {FIXED_BOX*1e3:.3f} pc  snaps {SNAP_START}–{SNAP_END}")
    print("=" * 70)
    start_time = time.time()
    if parallelize:
        with Pool(num_cores, maxtasksperchild=50) as pool:
            results = pool.map(collect_wrapper, args_list)
        for count, raw_com_, ranges in results:
            raw_com_dict[count] = raw_com_
            if ranges is not None:
                colorbar_ranges[count] = ranges
    else:
        for arg in args_list:
            count, raw_com_, ranges = collect_wrapper(arg)
            raw_com_dict[count] = raw_com_
            if ranges is not None:
                colorbar_ranges[count] = ranges
    total_t = time.time() - start_time
    print(f"Collection done in {total_t/60:.1f} min ({total_t/3600:.2f} hr)")
    save_colorbar_ranges(colorbar_ranges, save_path)
    save_com_positions(raw_com_dict, save_path)
    print("Done. Run with --mode plot to generate images.")

elif MODE == 'plot':
    print("=" * 70)
    print(f"MODE: PLOT — {total_frames} frames, {num_cores} cores")
    print(f"  box = {FIXED_BOX*1e3:.3f} pc  snaps {SNAP_START}–{SNAP_END}")
    print("=" * 70)
    interpolated_ranges = load_and_interpolate_colorbar_ranges(save_path, total_frames)
    smoothed_coms       = load_and_smooth_coms(save_path, total_frames, kernel_size=15)
    if smoothed_coms is None:
        print("Warning: com_positions.txt not found — using per-frame raw COM (may be jittery).")
    if interpolated_ranges is None:
        print("Error: colorbar_ranges.txt not found. Run collect mode first.")
    else:
        start_time = time.time()
        if parallelize:
            with Pool(num_cores, maxtasksperchild=50) as pool:
                pool.map(plot_wrapper, args_list)
        else:
            for arg in args_list:
                plot_wrapper(arg)
        total_t = time.time() - start_time
        print(f"Plotting done in {total_t/60:.1f} min ({total_t/3600:.2f} hr)")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
