#!/usr/bin/env python
"""
starforge_movies_simple.py: Create surface density maps for simulation snapshots in parallel

Usage: starforge_movies_simple.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir or snapshot directory? [default: False]
    --path=<path>                                       Path to the snapshot directory [default: ./]
    --save_path=<save_path>                             Path to save the images [default: ./movies/]
    --image_box_size=<image_box_size>                   Size of the image box (fraction of BoxSize) [default: 0.4]
    --snapnum_range=<snapnum_range>                     Range of snapshots to plot [default: 0,100]
    --parallel=<parallel>                               Should the script execute in parallel? [default: True]
    --num_cores=<num_cores>                             Number of processors to run on [default: 8]
    --resolution=<resolution>                           Resolution of surface density map [default: 1000]
    --vmin=<vmin>                                       Minimum value for colorbar [default: 1]
    --vmax=<vmax>                                       Maximum value for colorbar [default: 2000]
    --center_on_stars=<center_on_stars>                 Center on stars instead of box center [default: False]
    --cmap=<cmap>                                       Colormap name(s); comma-separate two to combine them [default: cet_fire]
    --cmap_split=<cmap_split>                           Fraction of first colormap when combining two (0-1) [default: 0.5]
    --no_scale_bar                                      Do not add a scale bar to the images
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from hybrid_sims_utils.read_snap import *
from constants import *

import sys
import os
# Import get_scale_bar_size from fif_movies (same directory)
_movies_dir = os.path.dirname(os.path.abspath(__file__))
if _movies_dir not in sys.path:
    sys.path.insert(0, _movies_dir)
from fif_movies import get_scale_bar_size

from docopt import docopt
import multiprocessing
from meshoid import Meshoid
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import colorcet as cc
import cmasher as cmr

import os
import numpy as np


def get_cmap(cmap_str, cmap_split=0.5):
    """
    Resolve a colormap name (or two comma-separated names) to a matplotlib colormap.
    Supports matplotlib and cmasher colormaps. If two names are given, combines them.
    """
    def _resolve(name):
        name = name.strip()
        # Try cmasher first (registered as 'cmr.<name>' with matplotlib)
        try:
            return matplotlib.colormaps[f'cmr.{name}']
        except KeyError:
            pass
        # Fall back to matplotlib (covers colorcet 'cet_*' etc.)
        return matplotlib.colormaps[name]

    names = [n.strip() for n in cmap_str.split(',')]
    if len(names) == 1:
        return _resolve(names[0])

    # Combine two colormaps at the requested split fraction
    cmap_a = _resolve(names[0])
    cmap_b = _resolve(names[1])
    n = 256
    split = int(np.clip(cmap_split, 0, 1) * n)
    colors_a = cmap_a(np.linspace(0, 1, split))
    colors_b = cmap_b(np.linspace(0, 1, n - split))
    combined_colors = np.vstack([colors_a, colors_b])
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        f'{names[0]}_{names[1]}', combined_colors)


def plot_surface_density(pdata, star_data, fire_star_data, snap_num, center,
                         image_box_size, save_path, resolution, vmin, vmax, cmap='cet_fire', no_scale_bar=False):
    """
    Create a surface density plot for a single snapshot
    
    Parameters:
    -----------
    pdata : dict
        Gas particle data
    star_data : dict or None
        STARFORGE star particle data
    fire_star_data : dict or None
        FIRE star particle data
    snap_num : int
        Snapshot number
    center : array
        Center coordinates for the plot
    image_box_size : float
        Size of the image box
    save_path : str
        Path to save the image
    resolution : int
        Resolution of the surface density map
    vmin, vmax : float
        Min and max values for colorbar
    """
    
    try:
        # Extract gas particle data
        pos = pdata["Coordinates"]
        mass = pdata["Masses"]
        hsml = pdata["SmoothingLength"]
        
        # Create Meshoid object
        M = Meshoid(pos, mass, hsml)
        
        # Set up coordinate grid
        min_pos = center - image_box_size / 2
        max_pos = center + image_box_size / 2
        X = np.linspace(min_pos[0], max_pos[0], resolution)
        Y = np.linspace(min_pos[1], max_pos[1], resolution)
        X, Y = np.meshgrid(X, Y, indexing='ij')
        
        # Create figure
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Calculate surface density
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m, center=center,
                                               size=image_box_size, res=resolution)
        
        # Plot surface density
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2,
                          norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                          cmap=cmap)
        
        # Plot stars if available
        if star_data is not None and 'Coordinates' in star_data and len(star_data['Coordinates']) > 0:
            star_coords = star_data['Coordinates']
            star_masses = star_data['Masses']
            # Scale marker size by mass
            marker_size = 20 * (star_masses / np.max(star_masses))
            ax.scatter(star_coords[:, 0], star_coords[:, 1], 
                      c='white', s=marker_size, alpha=0.8, edgecolors='white', linewidths=0.5)
        
        if fire_star_data is not None and 'Coordinates' in fire_star_data and len(fire_star_data['Coordinates']) > 0:
            fire_star_coords = fire_star_data['Coordinates']
            fire_star_masses = fire_star_data['Masses']
            # Scale marker size by mass
            marker_size = 100 * (fire_star_masses / np.max(fire_star_masses))
            ax.scatter(fire_star_coords[:, 0], fire_star_coords[:, 1], 
                      c='cyan', s=marker_size, alpha=0.8, edgecolors='white', linewidths=0.5)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim([min_pos[0], max_pos[0]])
        ax.set_ylim([min_pos[1], max_pos[1]])
        plt.xticks([])
        plt.yticks([])
        
        # Add scalebar — size chosen automatically by get_scale_bar_size (from fif_movies).
        # image_box_size is in pc (code units); function expects kpc, returns kpc.
        if not no_scale_bar:
            fontprops = fm.FontProperties(size=18)
            scale_bar_size_kpc, scale_bar_label = get_scale_bar_size(image_box_size * pc / kpc)
            scale_bar_size = scale_bar_size_kpc * kpc / pc  # convert back to pc for plot coordinates
            scalebar = AnchoredSizeBar(ax.transData,
                                        scale_bar_size, scale_bar_label, 'upper left',
                                        pad=1,
                                        color='white',
                                        frameon=False,
                                        size_vertical=scale_bar_size/100,
                                        fontproperties=fontprops)
            ax.add_artist(scalebar)
        
        plt.tight_layout()
        
        # Save figure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        snap_name = f'snapshot_{snap_num:04d}.png'
        plt.savefig(os.path.join(save_path, snap_name), dpi=300, facecolor='black')
        print(f"Saved: {os.path.join(save_path, snap_name)}")
        plt.close()
        
    except Exception as e:
        print(f"[EXCEPTION] Error in plot_surface_density for snap {snap_num}: {e}")
        import traceback
        traceback.print_exc()


def process_snapshot(args):
    """Process a single snapshot in parallel"""
    snap_num, path, snapdir, save_path, image_box_size, resolution, vmin, vmax, center_on_stars, cmap, no_scale_bar = args
    
    try:
        # Load snapshot data - path is the directory containing snapshots
        header, pdata, star_data, fire_star_data, refine_data, snapname = get_snap_data_hybrid(
            '', path, snap_num, snapshot_suffix='', snapdir=snapdir, refinement_tag=False)
        
        # Determine center
        if center_on_stars and star_data is not None and 'Coordinates' in star_data and len(star_data['Coordinates']) > 0:
            # Center on center of mass of stars
            star_coords = star_data['Coordinates']
            star_masses = star_data['Masses']
            center = np.average(star_coords, axis=0, weights=star_masses)
        elif center_on_stars and fire_star_data is not None and 'Coordinates' in fire_star_data and len(fire_star_data['Coordinates']) > 0:
            # Center on center of mass of FIRE stars
            fire_star_coords = fire_star_data['Coordinates']
            fire_star_masses = fire_star_data['Masses']
            center = np.average(fire_star_coords, axis=0, weights=fire_star_masses)
        else:
            # Center on box center
            box_size = header['BoxSize']
            center = np.array([box_size / 2, box_size / 2, box_size / 2])

        # Convert image_box_size from fraction to actual size
        actual_box_size = header['BoxSize'] * image_box_size
        
        # Create surface density plot
        plot_surface_density(pdata, star_data, fire_star_data, snap_num, center,
                            actual_box_size, save_path, resolution, vmin, vmax, cmap, no_scale_bar)
        
        print(f'Finished processing snapshot {snap_num}')
        
    except Exception as e:
        print(f"[EXCEPTION] Error processing snapshot {snap_num}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    args = docopt(__doc__)
    
    # Parse arguments
    parallel = convert_to_bool(args['--parallel'])
    path = args['--path']
    num_cores = int(args['--num_cores'])
    snapdir = convert_to_bool(args['--snapdir'])
    save_path = args['--save_path']
    image_box_size = float(args['--image_box_size'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    resolution = int(args['--resolution'])
    vmin = float(args['--vmin'])
    vmax = float(args['--vmax'])
    center_on_stars = convert_to_bool(args['--center_on_stars'])
    cmap_split = float(args['--cmap_split'])
    cmap = get_cmap(args['--cmap'], cmap_split)
    no_scale_bar = args['--no_scale_bar']

    # Support single snapshot (one integer) or a range (two integers)
    if snapnum_range.size == 1:
        snap_list = [int(snapnum_range[0])]
        range_str = str(snap_list[0])
    else:
        snap_list = list(range(int(snapnum_range[0]), int(snapnum_range[1]) + 1))
        range_str = f"{snapnum_range[0]} to {snapnum_range[1]}"

    print(f"Configuration:")
    print(f"  Snapshot path: {path}")
    print(f"  Save path: {save_path}")
    print(f"  Snapshot range: {range_str}")
    print(f"  Image box size: {image_box_size} (fraction of BoxSize)")
    print(f"  Resolution: {resolution}")
    print(f"  Colorbar range: {vmin} to {vmax}")
    print(f"  Center on stars: {center_on_stars}")
    print(f"  Parallel: {parallel} (cores: {num_cores})")

    # Prepare arguments for each snapshot
    snap_args = [(snap_num, path, snapdir, save_path, image_box_size,
                  resolution, vmin, vmax, center_on_stars, cmap, no_scale_bar)
                 for snap_num in snap_list]
    
    if parallel:
        print(f"\nProcessing {len(snap_args)} snapshots in parallel with {num_cores} cores...")
        pool = multiprocessing.Pool(processes=num_cores)
        pool.map(process_snapshot, snap_args)
        pool.close()
        pool.join()
    else:
        print(f"\nProcessing {len(snap_args)} snapshots sequentially...")
        for snap_arg in snap_args:
            process_snapshot(snap_arg)
    
    print("\nAll snapshots processed!")
