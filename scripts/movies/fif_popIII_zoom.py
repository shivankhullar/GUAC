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
from movie_utils.scale_bar import get_scale_bar_size
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


# ===========================================================================
# CONFIG
# ===========================================================================
path_full  = "/mnt/ceph/users/skhullar/projects/SFIRE/m12f/"
sim_full   = "output_jeans_refinement"
sim_cutout = "output_cutout"

save_path = "/mnt/home/skhullar/projects/SFIRE/m12f/movies/popIII_zoom/"

SNAP_FULL  = 27
SNAP_START = 28
SNAP_END   = 611

AU_TO_KPC = 1.0 / 206265000.0
EVOL_BOX  = 8000  * AU_TO_KPC   # 8000 AU in kpc ≈ 3.878e-5 kpc
ZOUT_BOX  = 0.3e-3               # 0.3 pc in kpc

# Cutover from full sim → cutout when the zoom box drops to this size
CUTOVER_KPC = 0.5   # 500 pc

# Disk angular-momentum search radius — must match make_disk_movie_frames.py defaults
R_SEARCH_KPC = 1e-5   # ≈ 2063 AU (disk scale)
R_MAX_KPC    = 1e-5

GAS_FIELDS = ['Masses', 'Coordinates', 'SmoothingLength', 'Velocities',
              'Density', 'InternalEnergy', 'ElectronAbundance', 'Temperature']
# ===========================================================================


# ===========================================================================
# Disk projection helpers (verbatim from make_disk_movie_frames.py)
# ===========================================================================

def find_center(pdata, stardata):
    if stardata and len(stardata.get('Masses', [])) > 0:
        return (np.sum(stardata['Coordinates'] * stardata['Masses'][:, None], axis=0)
                / np.sum(stardata['Masses']))
    return pdata['Coordinates'][np.argmax(pdata['Density'])]


def get_disk_axis(gas_pos_kpc, gas_vel_kms, gas_masses_Msun, r_search_kpc):
    dists = np.linalg.norm(gas_pos_kpc, axis=1)
    mask  = dists < r_search_kpc
    if mask.sum() < 4:
        return np.array([0., 0., 1.])
    pos_cm  = gas_pos_kpc[mask] * kpc
    vel_cms = gas_vel_kms[mask] * 1e5
    m_g     = gas_masses_Msun[mask] * Msun
    L       = np.sum(m_g[:, None] * np.cross(pos_cm, vel_cms), axis=0)
    L_mag   = np.linalg.norm(L)
    return L / L_mag if L_mag > 0 else np.array([0., 0., 1.])


def rotation_matrix_to_z(L_hat):
    z_hat = np.array([0., 0., 1.])
    v     = np.cross(L_hat, z_hat)
    s     = np.linalg.norm(v)
    c     = np.dot(L_hat, z_hat)
    if s > 1e-10:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * (1 - c) / s**2
    return np.eye(3) if c > 0 else -np.eye(3)


def compute_disk_orientation(pdata, stardata, r_search_kpc=R_SEARCH_KPC, r_max_kpc=R_MAX_KPC):
    """Exact port of identify_disk from make_disk_movie_frames.py.

    Pre-filters to r_local = max(r_max*5, r_search*2), subtracts mass-weighted
    COM velocity within r_search, then calls get_disk_axis on COM-subtracted vels.
    Returns (com_raw, rot) where (coords - com_raw) @ rot.T gives face-on coords.
    """
    com   = find_center(pdata, stardata)
    pos_c = pdata['Coordinates'] - com
    dists = np.linalg.norm(pos_c, axis=1)

    r_local      = max(r_max_kpc * 5, r_search_kpc * 2)
    local        = dists < r_local
    pos_local    = pos_c[local]
    vel_local    = pdata['Velocities'][local]
    masses_local = pdata['Masses'][local] * 1e10

    search = np.linalg.norm(pos_local, axis=1) < r_search_kpc
    if search.sum() > 0:
        com_vel = (np.sum(vel_local[search] * masses_local[search, None], axis=0)
                   / np.sum(masses_local[search]))
    else:
        com_vel = np.zeros(3)

    vel_com = vel_local - com_vel
    L_hat   = get_disk_axis(pos_local, vel_com, masses_local, r_search_kpc)
    rot     = rotation_matrix_to_z(L_hat)
    return com, rot


# ===========================================================================
# View rotation helpers
# ===========================================================================

def get_rotation_matrix(axis, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s,  c]])
    elif axis == 'y':
        return np.array([[ c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:   # z
        return np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])


def apply_view_rotation(coords, rot_deg):
    """Rotate about the y-axis by rot_deg degrees.

    0°   = face-on  (disk normal along z, looking down z)
    90°  = edge-on  (disk normal now along -x)
    180° = face-on  (upside-down, disk normal along -z)
    270° = edge-on  (disk normal along +x, mirrored)
    360° = face-on  (back to start)

    GridSurfaceDensity always projects along z, so this rotation changes
    what orientation of the disk is seen in the image plane.
    """
    return coords @ get_rotation_matrix('y', rot_deg).T


# ===========================================================================
# Frame sequence
# ===========================================================================
#
# Segment A (frames   0– 449, 450 frames): zoom  130 kpc → 8000 AU
#   A1 (full sim snap_027):  frames 0 .. n_full_zoom-1  (box > CUTOVER_KPC)
#   A2 (cutout snap_028):    frames n_full_zoom .. 449   (box ≤ CUTOVER_KPC)
# Segment B (frames 450–1033, 584 frames): evolution at 8000 AU, 1 snap/frame
# Segment C (frames 1034–1250, 217 frames): zoom-out 8000 AU → 0.3 pc + edge-on tilt
#
# At 25 fps: A ≈ 18 s | B ≈ 23.4 s | C ≈ 8.7 s

n_zoom = 450
n_evol = SNAP_END - SNAP_START + 1   # 584
n_zout = 217
total_frames = n_zoom + n_evol + n_zout   # 1251

# Box sizes
_zoom_boxes = np.logspace(np.log10(130), np.log10(EVOL_BOX), n_zoom)

# Find where the zoom crosses CUTOVER_KPC (full sim → cutout transition)
_trans = np.where(_zoom_boxes <= CUTOVER_KPC)[0]
n_full_zoom = int(_trans[0]) if len(_trans) > 0 else n_zoom
n_cut_zoom  = n_zoom - n_full_zoom

image_box_sizes = np.concatenate([
    _zoom_boxes,                                                           # A
    np.repeat(EVOL_BOX, n_evol),                                          # B
    np.logspace(np.log10(EVOL_BOX), np.log10(ZOUT_BOX), n_zout),         # C
])

snap_nums_per_frame = np.concatenate([
    np.repeat(SNAP_FULL,  n_full_zoom),        # A1: full sim
    np.repeat(SNAP_START, n_cut_zoom),         # A2: cutout snap_028
    np.arange(SNAP_START, SNAP_END + 1),       # B:  one per snap
    np.repeat(SNAP_END,   n_zout),             # C:  snap_611
]).astype(int)

sim_types_per_frame = (
    ['full']   * n_full_zoom +
    ['cutout'] * (n_cut_zoom + n_evol + n_zout)
)

# ------------------------------------------------------------------
# Y-axis rotation angles (degrees)
#   0°   = face-on, 90° = edge-on, 180° = face-on (inverted),
#   270° = edge-on (mirrored), 360° = face-on (full cycle)
#
#   0–350:    0° → 360°  (full rotation during zoom-in)
#   351–699:  0°         (hold face-on; 360° = 0°, seamless)
#   700–1033: 0° → 180°  (rotate during second half of evolution)
#   1034–1250: 180° → 270° (90° rotation to edge-on during zoom-out)
# ------------------------------------------------------------------
rot_degs = np.concatenate([
    np.linspace(0, 360, 351),     # frames 0–350    (351)
    np.zeros(99),                  # frames 351–449  ( 99) — rest of zoom-in
    np.zeros(584),                 # frames 450–1033 (584) — full evolution, face-on
    np.linspace(0, 90, 217),      # frames 1034–1250 (217) — tilt to edge-on
])  # total 1251 ✓


# ===========================================================================
# Per-frame data dispatch
# ===========================================================================

def get_frame_data(sim_type, snap_num):
    """Return (pdata, stardata, fire_stardata, com, time_Myr, redshift).

    'full' frames: return globals (zero I/O — inherited by forked workers).
    'cutout' frames: load snapshot fresh, compute per-snapshot disk orientation
                     with R_SEARCH_KPC = 1e-5 kpc (≈ 2000 AU, disk scale).
    """
    if sim_type == 'full':
        return g_pdata, g_stardata, g_fire_stardata, g_com, g_time_Myr, g_redshift

    hdr, pdata, stardata, fsd, _, _ = get_snap_data_hybrid(
        sim_cutout, path_full, snap_num,
        movie_tag=True, custom_gas_fields=GAS_FIELDS,
        snapshot_suffix='', snapdir=False, refinement_tag=False, full_tag=False)
    hdr, pdata, stardata, fsd = convert_units_to_physical(hdr, pdata, stardata, fsd)

    a        = hdr["Time"]
    redshift = 1.0 / a - 1.0
    time_Myr = convert_scale_factor_to_time(a, hdr)

    ref_star = stardata if (stardata and len(stardata.get('Masses', [])) > 0) else fsd
    com_raw, rot = compute_disk_orientation(pdata, ref_star)

    pdata['Coordinates'] = (pdata['Coordinates'] - com_raw) @ rot.T
    if stardata:
        stardata['Coordinates'] = (stardata['Coordinates'] - com_raw) @ rot.T
    if fsd:
        fsd['Coordinates'] = (fsd['Coordinates'] - com_raw) @ rot.T

    return pdata, stardata, fsd, np.zeros(3), time_Myr, redshift


# ===========================================================================
# Collect colorbar ranges
# ===========================================================================

def collect_colorbar_ranges(count, com, pdata, gas_dists, image_box_size,
                             rot_deg, res=1920):
    dist_cut_off = image_box_size * 2

    mask   = gas_dists < dist_cut_off
    pos    = pdata['Coordinates'][mask]
    masses = pdata['Masses'][mask]
    hsml   = pdata['SmoothingLength'][mask]

    new_coords = apply_view_rotation(pos, rot_deg)
    sigma_gas  = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)

    if np.any(np.isnan(sigma_gas)):
        print(f"Frame {count:04d}: NaN in sigma_gas — skipping.")
        return None

    res_16 = res / 16
    crop   = (slice(None), slice(int(3.5 * res_16), int(12.5 * res_16)))

    sigma_valid = sigma_gas[crop][sigma_gas[crop] > 0]
    if len(sigma_valid) == 0:
        print(f"Frame {count:04d}: empty sigma_gas crop — skipping.")
        return None

    ranges = {
        'sigma_gas_min': float(sigma_valid.min()),
        'sigma_gas_max': float(sigma_valid.max()),
    }
    print(f"Frame {count:04d}: box={image_box_size:.3e} kpc rot={rot_deg:.1f}° | "
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

    sf = len(frames) * 0.02

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
# Render frame
# ===========================================================================

def plot_image(count, com, pdata, gas_dists, stardata, fire_stardata,
               time_Myr, redshift, image_box_size, rot_deg,
               res=1920, save_path='./', colorbar_limits=None):

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    res_16       = res / 16
    dist_cut_off = image_box_size * 2
    crop_start   = int(3.5  * res_16)   # 420 px
    crop_end     = int(12.5 * res_16)   # 1500 px
    crop_slices  = (slice(None), slice(crop_start, crop_end))

    mask   = gas_dists < dist_cut_off
    pos    = pdata['Coordinates'][mask]
    masses = pdata['Masses'][mask]
    hsml   = pdata['SmoothingLength'][mask]

    new_coords = apply_view_rotation(pos, rot_deg)
    sigma_gas  = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)

    if colorbar_limits is not None:
        sigma_vmin = colorbar_limits['sigma_gas_min']
        sigma_vmax = colorbar_limits['sigma_gas_max']
    else:
        s_v = sigma_gas[crop_slices][sigma_gas[crop_slices] > 0]
        sigma_vmin = float(s_v.min()) if len(s_v) > 0 else 1e-10
        sigma_vmax = float(s_v.max()) if len(s_v) > 0 else 1e10

    sigma_vmin = max(float(sigma_vmin), 1e-30)
    sigma_vmax = max(float(sigma_vmax), sigma_vmin * 10)

    ax.imshow(sigma_gas[crop_slices].T, origin='lower',
              norm=LogNorm(vmin=sigma_vmin, vmax=sigma_vmax),
              cmap=cmr.eclipse, alpha=1.0)

    # Star overlay — yellow * markers sized by mass, matching make_disk_movie_frames.py
    # Only rendered at small scales (below CUTOVER_KPC = 0.5 kpc) where sinks exist.
    if image_box_size < CUTOVER_KPC:
        y_lo = (crop_start / res - 0.5) * image_box_size
        y_hi = (crop_end   / res - 0.5) * image_box_size
        for sdata in [stardata, fire_stardata]:
            if sdata is None or len(sdata.get('Coordinates', [])) == 0:
                continue
            sc      = apply_view_rotation(sdata['Coordinates'], rot_deg)
            masses  = sdata.get('Masses', np.ones(len(sc)) * 1e-10)
            in_view = ((np.abs(sc[:, 0]) < image_box_size / 2) &
                       (sc[:, 1] > y_lo) & (sc[:, 1] < y_hi))
            if in_view.sum() == 0:
                continue
            pix_x  = (sc[in_view, 0] + image_box_size / 2) / image_box_size * res
            pix_y  = (sc[in_view, 1] + image_box_size / 2) / image_box_size * res - crop_start
            # s ∝ log10(M_star / Msun), compresses the dynamic range so that
            # low-mass protostars remain visible alongside more massive ones.
            # Floor at log10(1 Msun) = 0 avoids negative sizes for sub-solar masses.
            s_vals = np.maximum(60.0 * np.log10(np.maximum(masses[in_view] * 1e10, 1.0)), 60.0)
            ax.scatter(pix_x, pix_y, s=s_vals, c='yellow', marker='*',
                       zorder=5, edgecolors='black', linewidths=0.5)

    plt.xticks([])
    plt.yticks([])

    # Scale bar
    scale_bar_size, scale_bar_text = get_scale_bar_size(image_box_size)
    scale_bar_pix = scale_bar_size * res / image_box_size
    fontprops     = fm.FontProperties(size=18)
    scalebar      = AnchoredSizeBar(ax.transData,
                                    scale_bar_pix, scale_bar_text, 'upper left',
                                    pad=1, color='white', frameon=False,
                                    size_vertical=scale_bar_pix / 100,
                                    fontproperties=fontprops)
    ax.add_artist(scalebar)

    # Time + redshift annotation (bottom-right; time never negative)
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
# Load snap_028 (cutout) first → compute disk orientation
# Then load snap_027 (full sim) → apply snap_028's rotation for smooth transition
# ===========================================================================

# ── Step 1: snap_028 — disk orientation ──────────────────────────────────────
print("Loading cutout snap_028 to compute disk orientation (r_search = 1e-5 kpc)...")
_hdr28, _pdata28, _sdata28, _fsd28, _, _ = get_snap_data_hybrid(
    sim_cutout, path_full, SNAP_START,
    movie_tag=True, custom_gas_fields=GAS_FIELDS,
    snapshot_suffix='', snapdir=False, refinement_tag=False, full_tag=False)
_hdr28, _pdata28, _sdata28, _fsd28 = convert_units_to_physical(
    _hdr28, _pdata28, _sdata28, _fsd28)

_a28           = _hdr28["Time"]
g_ref_time_Myr = convert_scale_factor_to_time(_a28, _hdr28)
print(f"snap_028 reference time: {g_ref_time_Myr:.6f} Myr")

_ref_star28 = _sdata28 if (_sdata28 and len(_sdata28.get('Masses', [])) > 0) else _fsd28
_com28, _rot28 = compute_disk_orientation(_pdata28, _ref_star28)
del _pdata28, _sdata28, _fsd28

# ── Step 2: snap_027 — center on densest point, apply snap_028 rotation ──────
print("Loading full sim snap_027...")
_hdr, g_pdata, g_stardata, g_fire_stardata, _, _ = get_snap_data_hybrid(
    sim_full, path_full, SNAP_FULL,
    movie_tag=True, custom_gas_fields=GAS_FIELDS,
    snapshot_suffix='', snapdir=False, refinement_tag=False, full_tag=False)
_hdr, g_pdata, g_stardata, g_fire_stardata = convert_units_to_physical(
    _hdr, g_pdata, g_stardata, g_fire_stardata)

_a_full    = _hdr["Time"]
g_redshift = 1.0 / _a_full - 1.0
g_time_Myr = convert_scale_factor_to_time(_a_full, _hdr)

_com27 = g_pdata['Coordinates'][np.argmax(g_pdata['Density'])]

g_pdata['Coordinates'] = (g_pdata['Coordinates'] - _com27) @ _rot28.T
if g_stardata:
    g_stardata['Coordinates']      = (g_stardata['Coordinates']      - _com27) @ _rot28.T
if g_fire_stardata:
    g_fire_stardata['Coordinates'] = (g_fire_stardata['Coordinates'] - _com27) @ _rot28.T

g_com = np.zeros(3)
print(f"snap_027 loaded: {len(g_pdata['Coordinates']):,} gas particles | "
      f"z = {g_redshift:.4f} | n_full_zoom = {n_full_zoom} frames")


# ===========================================================================
# Mode selection & parallelization
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['collect', 'plot'], default='plot')
_args = parser.parse_args()
MODE = _args.mode

parallelize = True
num_cores   = 128

start_time      = time.time()
completed_count = 0
colorbar_ranges = {}
interpolated_ranges = None


def collect_wrapper(args):
    global completed_count, start_time
    count, snap_num, sim_type, image_box_size, rot_deg = args

    pdata_, sdata_, fsd_, com_, time_Myr_, redshift_ = get_frame_data(sim_type, snap_num)
    gas_dists_ = np.linalg.norm(pdata_['Coordinates'], axis=1)

    ranges = collect_colorbar_ranges(count, com_, pdata_, gas_dists_,
                                     image_box_size, rot_deg, res=1920)
    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_t   = elapsed / completed_count
        remain  = total_frames - completed_count
        print(f"Collect {completed_count}/{total_frames} | "
              f"{elapsed/60:.1f} min | ~{avg_t*remain/60:.1f} min left")
    return count, ranges


def plot_wrapper(args):
    global completed_count, start_time, interpolated_ranges
    count, snap_num, sim_type, image_box_size, rot_deg = args

    pdata_, sdata_, fsd_, com_, time_Myr_, redshift_ = get_frame_data(sim_type, snap_num)
    gas_dists_ = np.linalg.norm(pdata_['Coordinates'], axis=1)

    colorbar_limits = interpolated_ranges[count] if interpolated_ranges is not None else None
    plot_image(count, com_, pdata_, gas_dists_, sdata_, fsd_,
               time_Myr_, redshift_, image_box_size, rot_deg,
               res=1920, save_path=save_path, colorbar_limits=colorbar_limits)

    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_t   = elapsed / completed_count
        remain  = total_frames - completed_count
        print(f"Plot {completed_count}/{total_frames} | "
              f"{elapsed/60:.1f} min | ~{avg_t*remain/60:.1f} min left")


# ===========================================================================
# Main
# ===========================================================================
os.makedirs(save_path, exist_ok=True)

args_list = [
    (i, int(snap_nums_per_frame[i]), sim_types_per_frame[i],
     float(image_box_sizes[i]), float(rot_degs[i]))
    for i in range(total_frames)
]

if MODE == 'collect':
    print("=" * 70)
    print(f"MODE: COLLECT — {total_frames} frames, {num_cores} cores")
    print(f"  n_full_zoom={n_full_zoom}  n_cut_zoom={n_cut_zoom}  "
          f"n_evol={n_evol}  n_zout={n_zout}")
    print("=" * 70)
    start_time = time.time()
    if parallelize:
        with Pool(num_cores, maxtasksperchild=50) as pool:
            results = pool.map(collect_wrapper, args_list)
        for count, ranges in results:
            if ranges is not None:
                colorbar_ranges[count] = ranges
    else:
        for arg in args_list:
            count, ranges = collect_wrapper(arg)
            if ranges is not None:
                colorbar_ranges[count] = ranges
    total_t = time.time() - start_time
    print(f"Collection done in {total_t/60:.1f} min ({total_t/3600:.2f} hr)")
    save_colorbar_ranges(colorbar_ranges, save_path)
    print("Done. Run with --mode plot to generate images.")

elif MODE == 'plot':
    print("=" * 70)
    print(f"MODE: PLOT — {total_frames} frames, {num_cores} cores")
    print("=" * 70)
    interpolated_ranges = load_and_interpolate_colorbar_ranges(save_path, total_frames)
    if interpolated_ranges is None:
        print("Error: run collect mode first.")
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
