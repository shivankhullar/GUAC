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
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from hybrid_sims_utils.read_snap import *

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

import os
import numpy as np


def plot_surface_density(pdata, star_data, fire_star_data, snap_num, center, 
                         image_box_size, save_path, resolution, vmin, vmax):
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
                          cmap='cet_fire')
        
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
        
        # Add scalebar
        fontprops = fm.FontProperties(size=18)
        scale_bar_size = float(image_box_size / 4)
        scalebar = AnchoredSizeBar(ax.transData,
                                    scale_bar_size, '%.1f pc'%(scale_bar_size), 'upper left',
                                    pad=1,
                                    color='white',
                                    frameon=False,
                                    size_vertical=0.1,
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
    snap_num, path, snapdir, save_path, image_box_size, resolution, vmin, vmax, center_on_stars = args
    
    try:
        # Load snapshot data - path is the directory containing snapshots
        pdata, star_data, fire_star_data, refine_data, snapname = get_snap_data_hybrid(
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
            box_size = pdata['BoxSize']
            center = np.array([box_size / 2, box_size / 2, box_size / 2])
        
        # Convert image_box_size from fraction to actual size
        actual_box_size = pdata['BoxSize'] * image_box_size
        
        # Create surface density plot
        plot_surface_density(pdata, star_data, fire_star_data, snap_num, center,
                            actual_box_size, save_path, resolution, vmin, vmax)
        
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
    
    print(f"Configuration:")
    print(f"  Snapshot path: {path}")
    print(f"  Save path: {save_path}")
    print(f"  Snapshot range: {snapnum_range[0]} to {snapnum_range[1]}")
    print(f"  Image box size: {image_box_size} (fraction of BoxSize)")
    print(f"  Resolution: {resolution}")
    print(f"  Colorbar range: {vmin} to {vmax}")
    print(f"  Center on stars: {center_on_stars}")
    print(f"  Parallel: {parallel} (cores: {num_cores})")
    
    # Prepare arguments for each snapshot
    snap_args = [(snap_num, path, snapdir, save_path, image_box_size, 
                  resolution, vmin, vmax, center_on_stars) 
                 for snap_num in range(snapnum_range[0], snapnum_range[1] + 1)]
    
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
