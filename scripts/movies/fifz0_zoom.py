#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']     = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']     = '1'

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from hybrid_sims_utils.read_snap import *
from hybrid_sims_utils.hybrid_utils import *
from movie_utils.scale_bar import get_scale_bar_size
from constants import *

from meshoid import Meshoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
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
from tqdm import tqdm
import pickle
import itertools
from multiprocessing import Pool
import functools
import time
from scipy.interpolate import interp1d, UnivariateSpline


# ===========================================================================
# CONFIG — edit these to change simulation / output
# ===========================================================================
path      = "/mnt/ceph/users/skhullar/projects/SFIRE/z0_m12f_tests/"
sim       = "output_max_dense_early_jeans"
snap_num  = 164
save_path = "/mnt/home/skhullar/projects/SFIRE/z0_m12f_tests/movies/z0_zoom/"

snapshot_suffix = ""
snapdir         = False
refinement_tag  = False
full_tag        = False
movie_tag       = True

# DM config
# DM is shown at large scales and fades out as we zoom in.
# The transition is linear between DM_ALPHA_START_KPC and DM_ALPHA_END_KPC.
# With the default zoom sequence this spans ~29 frames (~1 sec at 25 fps).
USE_DM1            = True   # hi-res DM (PartType1)
USE_DM2            = True   # low-res DM (PartType2)
DM_ALPHA_START_KPC = 1500   # box size at which DM begins fading out
DM_ALPHA_END_KPC   = 750    # box size at which DM is fully invisible
DM1_MAX_ALPHA      = 0.5    # peak alpha for PartType1 layer
DM2_MAX_ALPHA      = 0.3    # peak alpha for PartType2 layer (lower — fewer, heavier particles)
# ===========================================================================


def get_rotation_matrix(axis='x', angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis.lower() == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]])
    elif axis.lower() == 'y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])
    elif axis.lower() == 'z':
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def rotate_coordinates(coords, axis='x', angle_deg=90, center=None):
    R = get_rotation_matrix(axis, angle_deg)
    if center is not None:
        coords_centered = coords - center
        return np.dot(coords_centered, R.T) + center
    else:
        return np.dot(coords, R.T)




def get_layer_alphas(image_box_size):
    """Return (dm1_alpha, dm2_alpha, sigma_gas_alpha, temp_gas_alpha) for the given box size.

    At large scales (>DM_ALPHA_START_KPC): DM at full alpha, gas at reduced alpha.
    At small scales (<DM_ALPHA_END_KPC): no DM, gas at full alpha.
    Linear transition in between.
    """
    if image_box_size >= DM_ALPHA_START_KPC:
        t = 0.0
    elif image_box_size <= DM_ALPHA_END_KPC:
        t = 1.0
    else:
        t = (DM_ALPHA_START_KPC - image_box_size) / (DM_ALPHA_START_KPC - DM_ALPHA_END_KPC)

    dm1_alpha   = DM1_MAX_ALPHA * (1.0 - t)
    dm2_alpha   = DM2_MAX_ALPHA * (1.0 - t)
    sigma_alpha = 0.5 + 0.3 * t   # 0.5 → 0.8
    temp_alpha  = 0.3 + 0.2 * t   # 0.3 → 0.5

    return dm1_alpha, dm2_alpha, sigma_alpha, temp_alpha


def collect_colorbar_ranges(count, com, pdata, image_box_size, gas_dists, axis, angle_deg, res=1920):
    """Collect per-frame colorbar min/max for gas and DM without saving images."""
    dist_cut_off = image_box_size * 2

    pos    = pdata["Coordinates"][gas_dists < dist_cut_off]
    masses = pdata["Masses"][gas_dists < dist_cut_off]
    hsml   = pdata["SmoothingLength"][gas_dists < dist_cut_off]
    temps  = pdata["Temperature"][gas_dists < dist_cut_off]

    new_coords = rotate_coordinates(pos, axis=axis, angle_deg=angle_deg, center=com)

    sigma_gas = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)
    temp_gas  = GridSurfaceDensity(masses * temps, new_coords, hsml, com, image_box_size, res=res) / sigma_gas

    res_16 = res / 16
    crop   = (slice(None), slice(int(3.5 * res_16), int(12.5 * res_16)))

    sigma_valid = sigma_gas[crop][sigma_gas[crop] > 0]
    temp_valid  = temp_gas[crop][temp_gas[crop]   > 0]

    ranges = {
        'sigma_gas_min': sigma_valid.min() if len(sigma_valid) > 0 else 1e-10,
        'sigma_gas_max': sigma_valid.max() if len(sigma_valid) > 0 else 1e10,
        'temp_gas_min':  temp_valid.min()  if len(temp_valid)  > 0 else 1e1,
        'temp_gas_max':  temp_valid.max()  if len(temp_valid)  > 0 else 1e8,
        'dm1_min': 1e-10, 'dm1_max': 1e-2,   # defaults; overwritten below if DM is computed
        'dm2_min': 1e-10, 'dm2_max': 1e-2,
    }

    # Compute DM ranges only when the DM layer will actually be visible
    if image_box_size > DM_ALPHA_END_KPC:
        if USE_DM1 and dm1_pos is not None:
            dm1_mask = dm1_dists < dist_cut_off
            if dm1_mask.sum() > 0:
                new_dm1 = rotate_coordinates(dm1_pos[dm1_mask], axis=axis, angle_deg=angle_deg, center=com)
                s_dm1   = GridSurfaceDensity(dm1_mass[dm1_mask], new_dm1, dm1_hsml[dm1_mask],
                                             com, image_box_size, res=res)
                v = s_dm1[crop][s_dm1[crop] > 0]
                if len(v) > 0:
                    ranges['dm1_min'] = v.min()
                    ranges['dm1_max'] = v.max()

        if USE_DM2 and dm2_pos is not None:
            dm2_mask = dm2_dists < dist_cut_off
            if dm2_mask.sum() > 0:
                new_dm2 = rotate_coordinates(dm2_pos[dm2_mask], axis=axis, angle_deg=angle_deg, center=com)
                s_dm2   = GridSurfaceDensity(dm2_mass[dm2_mask], new_dm2, dm2_hsml[dm2_mask],
                                             com, image_box_size, res=res)
                v = s_dm2[crop][s_dm2[crop] > 0]
                if len(v) > 0:
                    ranges['dm2_min'] = v.min()
                    ranges['dm2_max'] = v.max()

    print(f"Frame {count:04d}: box={image_box_size:.2e} kpc | "
          f"sigma=[{ranges['sigma_gas_min']:.2e}, {ranges['sigma_gas_max']:.2e}] | "
          f"temp=[{ranges['temp_gas_min']:.2e}, {ranges['temp_gas_max']:.2e}] | "
          f"dm1=[{ranges['dm1_min']:.2e}, {ranges['dm1_max']:.2e}]")

    return ranges


def save_colorbar_ranges(ranges_dict, save_path):
    """Save colorbar ranges to text file."""
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')
    with open(ranges_file, 'w') as f:
        f.write("# Frame sigma_gas_min sigma_gas_max temp_gas_min temp_gas_max "
                "dm1_min dm1_max dm2_min dm2_max\n")
        for frame in sorted(ranges_dict.keys()):
            r = ranges_dict[frame]
            f.write(f"{frame} "
                    f"{r['sigma_gas_min']:.10e} {r['sigma_gas_max']:.10e} "
                    f"{r['temp_gas_min']:.10e} {r['temp_gas_max']:.10e} "
                    f"{r['dm1_min']:.10e} {r['dm1_max']:.10e} "
                    f"{r['dm2_min']:.10e} {r['dm2_max']:.10e}\n")
    print(f"Saved colorbar ranges to {ranges_file}")


def load_and_interpolate_colorbar_ranges(save_path, num_frames):
    """Load colorbar ranges and interpolate smoothly with cubic splines."""
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')

    if not os.path.exists(ranges_file):
        print(f"Error: {ranges_file} not found. Run in 'collect' mode first.")
        return None

    data = np.loadtxt(ranges_file)
    frames        = data[:, 0].astype(int)
    sigma_gas_min = data[:, 1]
    sigma_gas_max = data[:, 2]
    temp_gas_min  = data[:, 3]
    temp_gas_max  = data[:, 4]
    dm1_min       = data[:, 5]
    dm1_max       = data[:, 6]
    dm2_min       = data[:, 7]
    dm2_max       = data[:, 8]

    sf_gas = len(frames) * 0.02    # increased for smoother colorbar transitions
    sf_dm  = len(frames) * 0.01

    interp_sigma_min = UnivariateSpline(frames, np.log10(sigma_gas_min), k=3, s=sf_gas)
    interp_sigma_max = UnivariateSpline(frames, np.log10(sigma_gas_max), k=3, s=sf_gas)
    interp_temp_min  = UnivariateSpline(frames, np.log10(temp_gas_min),  k=3, s=sf_dm)
    interp_temp_max  = UnivariateSpline(frames, np.log10(temp_gas_max),  k=3, s=sf_dm)
    interp_dm1_min   = UnivariateSpline(frames, np.log10(dm1_min),       k=3, s=sf_dm)
    interp_dm1_max   = UnivariateSpline(frames, np.log10(dm1_max),       k=3, s=sf_dm)
    interp_dm2_min   = UnivariateSpline(frames, np.log10(dm2_min),       k=3, s=sf_dm)
    interp_dm2_max   = UnivariateSpline(frames, np.log10(dm2_max),       k=3, s=sf_dm)

    all_frames = np.arange(num_frames)
    interpolated_ranges = {}
    for frame in all_frames:
        interpolated_ranges[frame] = {
            'sigma_gas_min': 10 ** interp_sigma_min(frame),
            'sigma_gas_max': 10 ** interp_sigma_max(frame),
            'temp_gas_min':  10 ** interp_temp_min(frame),
            'temp_gas_max':  10 ** interp_temp_max(frame),
            'dm1_min':       10 ** interp_dm1_min(frame),
            'dm1_max':       10 ** interp_dm1_max(frame),
            'dm2_min':       10 ** interp_dm2_min(frame),
            'dm2_max':       10 ** interp_dm2_max(frame),
        }

    print(f"Loaded and smoothed colorbar ranges for {num_frames} frames using cubic splines")
    return interpolated_ranges


def plot_image(count, com, pdata, image_box_size, gas_dists, axis, angle_deg,
               res=1920, save_path='./', colorbar_limits=None):
    """Render a single frame and save to disk."""
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    res_16       = res / 16
    dist_cut_off = image_box_size * 2
    crop         = (slice(None), slice(int(3.5 * res_16), int(12.5 * res_16)))

    # Gas projections
    pos    = pdata["Coordinates"][gas_dists < dist_cut_off]
    masses = pdata["Masses"][gas_dists < dist_cut_off]
    hsml   = pdata["SmoothingLength"][gas_dists < dist_cut_off]
    temps  = pdata["Temperature"][gas_dists < dist_cut_off]

    new_coords = rotate_coordinates(pos, axis=axis, angle_deg=angle_deg, center=com)

    sigma_gas = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)
    temp_gas  = GridSurfaceDensity(masses * temps, new_coords, hsml, com, image_box_size, res=res) / sigma_gas

    # Colorbar limits
    if colorbar_limits is not None:
        sigma_vmin = colorbar_limits['sigma_gas_min']
        sigma_vmax = colorbar_limits['sigma_gas_max']
        temp_vmin  = colorbar_limits['temp_gas_min']
        temp_vmax  = colorbar_limits['temp_gas_max']
        dm1_vmin   = colorbar_limits['dm1_min']
        dm1_vmax   = colorbar_limits['dm1_max']
        dm2_vmin   = colorbar_limits['dm2_min']
        dm2_vmax   = colorbar_limits['dm2_max']
    else:
        sigma_valid = sigma_gas[crop][sigma_gas[crop] > 0]
        temp_valid  = temp_gas[crop][temp_gas[crop]   > 0]
        sigma_vmin  = sigma_valid.min() if len(sigma_valid) > 0 else 1e-10
        sigma_vmax  = sigma_valid.max() if len(sigma_valid) > 0 else 1e10
        temp_vmin   = temp_valid.min()  if len(temp_valid)  > 0 else 1e1
        temp_vmax   = temp_valid.max()  if len(temp_valid)  > 0 else 1e8
        dm1_vmin = dm2_vmin = 1e-10
        dm1_vmax = dm2_vmax = 1e-2

    # Layer alphas — transition from DM-dominant to gas-only
    dm1_alpha, dm2_alpha, sigma_alpha, temp_alpha = get_layer_alphas(image_box_size)

    # Render: gas layers first, then DM on top
    ax.imshow(sigma_gas[crop].T, origin='lower',
              norm=LogNorm(vmin=sigma_vmin, vmax=sigma_vmax),
              cmap=cmr.eclipse_r, alpha=sigma_alpha)
    ax.imshow(temp_gas[crop].T, origin='lower',
              norm=LogNorm(vmin=temp_vmin, vmax=temp_vmax),
              cmap=cmr.eclipse, alpha=temp_alpha)

    # DM layers (only computed and rendered when alpha > 0)
    if dm1_alpha > 0 and USE_DM1 and dm1_pos is not None:
        dm1_mask = dm1_dists < dist_cut_off
        if dm1_mask.sum() > 0:
            new_dm1   = rotate_coordinates(dm1_pos[dm1_mask], axis=axis, angle_deg=angle_deg, center=com)
            sigma_dm1 = GridSurfaceDensity(dm1_mass[dm1_mask], new_dm1, dm1_hsml[dm1_mask],
                                           com, image_box_size, res=res)
            ax.imshow(sigma_dm1[crop].T, origin='lower',
                      norm=LogNorm(vmin=dm1_vmin, vmax=dm1_vmax),
                      cmap='inferno', alpha=dm1_alpha)

    if dm2_alpha > 0 and USE_DM2 and dm2_pos is not None:
        dm2_mask = dm2_dists < dist_cut_off
        if dm2_mask.sum() > 0:
            new_dm2   = rotate_coordinates(dm2_pos[dm2_mask], axis=axis, angle_deg=angle_deg, center=com)
            sigma_dm2 = GridSurfaceDensity(dm2_mass[dm2_mask], new_dm2, dm2_hsml[dm2_mask],
                                           com, image_box_size, res=res)
            ax.imshow(sigma_dm2[crop].T, origin='lower',
                      norm=LogNorm(vmin=dm2_vmin, vmax=dm2_vmax),
                      cmap='hot', alpha=dm2_alpha)

    plt.xticks([])
    plt.yticks([])

    scale_bar_size, scale_bar_text = get_scale_bar_size(image_box_size)
    scale_bar_size_pixels = scale_bar_size * res / image_box_size

    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                               scale_bar_size_pixels, scale_bar_text, 'upper left',
                               pad=1,
                               color='white',
                               frameon=False,
                               size_vertical=scale_bar_size_pixels / 100,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)
    fig.set_facecolor('black')

    plt.tight_layout()

    image_save_path = save_path + 'surf_dens/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    save_file = f"snap_{count:04d}.png"
    plt.savefig(image_save_path + save_file, dpi=250)
    plt.close(fig)
    print(f"Saved {save_file} to {image_save_path}")


# ===========================================================================
# Load gas data (once)
# ===========================================================================
print("Loading gas data...")

header, pdata, stardata, fire_stardata, _, _ = get_snap_data_hybrid(
    sim, path, snap_num,
    snapshot_suffix=snapshot_suffix,
    snapdir=snapdir,
    refinement_tag=refinement_tag,
    full_tag=full_tag,
    movie_tag=movie_tag)

print("Loaded data. Converting units...")

header, pdata, stardata, fire_stardata = convert_units_to_physical(header, pdata, stardata, fire_stardata)

# Apply projection matrix in-place to put galaxy in standard orientation (edge-on after proj).
# Per-frame rotation: angle=90° around x → face-on; angle=0° → edge-on.
print("Applying galaxy projection matrix...")
gal_proj_matrix = get_hybrid_galaxy_proj_matrix(sim, path, snap_num)
header["RefinementRegionCenter"] = gal_proj_matrix @ header["RefinementRegionCenter"]
pdata["Coordinates"]             = pdata["Coordinates"]             @ gal_proj_matrix.T
if stardata:
    stardata["Coordinates"]      = stardata["Coordinates"]          @ gal_proj_matrix.T
if fire_stardata:
    fire_stardata["Coordinates"] = fire_stardata["Coordinates"]     @ gal_proj_matrix.T

com       = header["RefinementRegionCenter"]
gas_dists = np.linalg.norm(pdata["Coordinates"] - com, axis=1)

print(f"Galaxy center: {com}")
print(f"Total gas particles: {len(gas_dists)}")


# ===========================================================================
# Load DM data (once) — cached to disk to avoid recomputing Meshoid hsml
# ===========================================================================
dm1_pos = dm1_mass = dm1_hsml = dm1_dists = None
dm2_pos = dm2_mass = dm2_hsml = dm2_dists = None

dm_pickle_dir  = "/mnt/home/skhullar/projects/SFIRE/z0_m12f_tests/movies/dm_data/"
dm_pickle_path = os.path.join(dm_pickle_dir, f"snapshot_{snap_num:03d}.pkl")

if os.path.exists(dm_pickle_path):
    print(f"Loading DM data from cache: {dm_pickle_path}")
    with open(dm_pickle_path, 'rb') as f:
        dm_cache = pickle.load(f)
    dm1_pos   = dm_cache.get('dm1_pos')
    dm1_mass  = dm_cache.get('dm1_mass')
    dm1_hsml  = dm_cache.get('dm1_hsml')
    dm1_dists = dm_cache.get('dm1_dists')
    dm2_pos   = dm_cache.get('dm2_pos')
    dm2_mass  = dm_cache.get('dm2_mass')
    dm2_hsml  = dm_cache.get('dm2_hsml')
    dm2_dists = dm_cache.get('dm2_dists')
    if dm1_pos is not None:
        print(f"DM1: {len(dm1_mass):,} particles loaded from cache.")
    if dm2_pos is not None:
        print(f"DM2: {len(dm2_mass):,} particles loaded from cache.")

else:
    print(f"No DM cache found — computing from snapshot and saving to {dm_pickle_path}")
    snap_file = os.path.join(path, sim, f"snapshot_{snap_num:03d}.hdf5")
    dm_cache  = {}

    with h5py.File(snap_file, 'r') as f:
        a = header["Time"]
        h = header["HubbleParam"]

        if USE_DM1 and "PartType1" in f:
            _coords = convert_quant_to_physical(f["PartType1"]["Coordinates"][:],
                                                key="Coordinates", a=a, h=h)
            _mass   = convert_quant_to_physical(f["PartType1"]["Masses"][:],
                                                key="Masses", a=a, h=h)
            dm1_pos   = _coords @ gal_proj_matrix.T
            dm1_mass  = _mass
            dm1_dists = np.linalg.norm(dm1_pos - com, axis=1)
            print(f"DM1: {len(dm1_mass):,} particles — computing smoothing lengths...")
            dm1_hsml  = Meshoid(dm1_pos, dm1_mass).SmoothingLength()
            print(f"DM1 smoothing lengths done.")
            dm_cache.update({'dm1_pos': dm1_pos, 'dm1_mass': dm1_mass,
                             'dm1_hsml': dm1_hsml, 'dm1_dists': dm1_dists})

        if USE_DM2 and "PartType2" in f:
            _coords = convert_quant_to_physical(f["PartType2"]["Coordinates"][:],
                                                key="Coordinates", a=a, h=h)
            _mass   = convert_quant_to_physical(f["PartType2"]["Masses"][:],
                                                key="Masses", a=a, h=h)
            dm2_pos   = _coords @ gal_proj_matrix.T
            dm2_mass  = _mass
            dm2_dists = np.linalg.norm(dm2_pos - com, axis=1)
            print(f"DM2: {len(dm2_mass):,} particles — computing smoothing lengths...")
            dm2_hsml  = Meshoid(dm2_pos, dm2_mass).SmoothingLength()
            print(f"DM2 smoothing lengths done.")
            dm_cache.update({'dm2_pos': dm2_pos, 'dm2_mass': dm2_mass,
                             'dm2_hsml': dm2_hsml, 'dm2_dists': dm2_dists})

    os.makedirs(dm_pickle_dir, exist_ok=True)
    with open(dm_pickle_path, 'wb') as f:
        pickle.dump(dm_cache, f)
    print(f"DM data saved to {dm_pickle_path}")


# ===========================================================================
# Zoom & rotation sequences  ← tune these to change movie timing/length
# ===========================================================================
# 1250 frames total.  All rotations around x-axis.
# gal_proj_matrix gives face-on at angle=0°.  Edge-on = +90° around x.
#
# Segment A    0– 350 (350 f): zoom in  3.6 Mpc → 500 pc,  edge-on (90°)
# Segment B  350– 550 (200 f): zoom in  500 pc  →   1 pc,  edge-on (90°)
# Segment C  550– 650 (100 f): zoom out   1 pc  → 250 pc,  rotation begins
# Segment D  650– 950 (300 f): hold at 250 pc,             rotation continues
# Segment E  950–1250 (300 f): zoom out 250 pc  → 3.6 Mpc, face-on then re-tilts
#
# Rotation 1: frames 550–950 (400 f), 90°→360°  (back to face-on)
# Rotation 2: frames 1100–1250 (150 f), 0°→90° (returns to starting edge-on)

image_box_sizes = np.concatenate([
    np.logspace(np.log10(3600),  np.log10(0.5),    350),  # A: 3.6 Mpc → 500 pc
    np.logspace(np.log10(0.5),   np.log10(0.001),  200),  # B: 500 pc → 1 pc
    np.logspace(np.log10(0.001), np.log10(0.25),   100),  # C: 1 pc → 250 pc
    np.repeat(0.25,                                300),   # D: hold at 250 pc
    np.logspace(np.log10(0.25),  np.log10(3600),   300),  # E: 250 pc → 3.6 Mpc
])

axes = ["x"] * len(image_box_sizes)
angle_degs = np.concatenate([
    np.repeat(90.0,             550),   # A+B: edge-on (0–550)
    np.linspace(90.0, 360.0,   400),   # C+D: rotate 90°→360° (550–950)
    np.repeat(360.0,            150),   # E start: face-on (950–1100)
    np.linspace(0.0, 90.0,     150),   # E end: re-tilt to edge-on (1100–1250)
])


# ===========================================================================
# Mode selection & parallelization
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['collect', 'plot'], default='plot',
                    help="'collect' to gather colorbar ranges, 'plot' to create images")
_args = parser.parse_args()
MODE = _args.mode

parallelize = True
num_cores   = 128

start_time          = time.time()
completed_count     = 0
colorbar_ranges     = {}
interpolated_ranges = None


def collect_wrapper(args):
    global completed_count, start_time, colorbar_ranges
    count, image_box_size = args
    ranges = collect_colorbar_ranges(count, com, pdata, image_box_size, gas_dists,
                                     axes[count], angle_degs[count], res=1920)
    colorbar_ranges[count] = ranges

    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / completed_count
        remaining = len(image_box_sizes) - completed_count
        print(f"Collecting Progress: {completed_count}/{len(image_box_sizes)} | "
              f"Elapsed: {elapsed/60:.1f}min | Est. remaining: {avg_time*remaining/60:.1f}min")

    return count, ranges


def plot_wrapper(args):
    global completed_count, start_time, interpolated_ranges
    count, image_box_size = args
    colorbar_limits = interpolated_ranges[count] if interpolated_ranges is not None else None
    plot_image(count, com, pdata, image_box_size, gas_dists,
               axes[count], angle_degs[count], res=1920,
               save_path=save_path, colorbar_limits=colorbar_limits)

    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / completed_count
        remaining = len(image_box_sizes) - completed_count
        print(f"Plotting Progress: {completed_count}/{len(image_box_sizes)} | "
              f"Elapsed: {elapsed/60:.1f}min | Est. remaining: {avg_time*remaining/60:.1f}min")


# ===========================================================================
# Main
# ===========================================================================
if not os.path.exists(save_path):
    os.makedirs(save_path)

args_list = [(i, box_size) for i, box_size in enumerate(image_box_sizes)]

if MODE == 'collect':
    print("=" * 70)
    print("MODE: COLLECT — gathering colorbar ranges from all frames")
    print("=" * 70)

    start_time = time.time()
    if parallelize:
        print(f"Using {num_cores} cores | {len(image_box_sizes)} frames")
        with Pool(num_cores, maxtasksperchild=50) as pool:
            results = pool.map(collect_wrapper, args_list)
        for count, ranges in results:
            colorbar_ranges[count] = ranges
    else:
        for count, image_box_size in enumerate(image_box_sizes):
            ranges = collect_colorbar_ranges(count, com, pdata, image_box_size, gas_dists,
                                             axes[count], angle_degs[count], res=1920)
            colorbar_ranges[count] = ranges

    total_time = time.time() - start_time
    print(f"Collection time: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
    save_colorbar_ranges(colorbar_ranges, save_path)
    print("Done. Run with --mode plot to generate images.")

elif MODE == 'plot':
    print("=" * 70)
    print("MODE: PLOT — creating images with smoothly interpolated colorbars")
    print("=" * 70)

    interpolated_ranges = load_and_interpolate_colorbar_ranges(save_path, len(image_box_sizes))

    if interpolated_ranges is None:
        print("Error: could not load colorbar ranges. Exiting.")
    else:
        start_time = time.time()
        if parallelize:
            print(f"Using {num_cores} cores | {len(image_box_sizes)} frames")
            with Pool(num_cores, maxtasksperchild=50) as pool:
                pool.map(plot_wrapper, args_list)
        else:
            for count, image_box_size in enumerate(image_box_sizes):
                colorbar_limits = interpolated_ranges[count]
                plot_image(count, com, pdata, image_box_size, gas_dists,
                           axes[count], angle_degs[count], res=1920,
                           save_path=save_path, colorbar_limits=colorbar_limits)

        total_time = time.time() - start_time
        print(f"Total plotting time: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")

else:
    print(f"Error: invalid MODE '{MODE}'. Must be 'collect' or 'plot'.")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
