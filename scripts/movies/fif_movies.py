
from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from hybrid_sims_utils.read_snap import *
from constants import *

from meshoid import Meshoid
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import colorcet as cc
#%matplotlib inline

import glob
import re
import os
from tqdm import tqdm
import pickle
import itertools
from multiprocessing import Pool
import functools
import time



def get_rotation_matrix(axis='x', angle_deg=90):
    """
    Get rotation matrix for rotating coordinates around a given axis.
    
    Parameters:
    -----------
    axis : str
        Axis to rotate around ('x', 'y', or 'z')
    angle_deg : float
        Rotation angle in degrees
    
    Returns:
    --------
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    if axis.lower() == 'x':
        # Rotation around x-axis
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    elif axis.lower() == 'y':
        # Rotation around y-axis
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    elif axis.lower() == 'z':
        # Rotation around z-axis
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def rotate_coordinates(coords, axis='x', angle_deg=90, center=None):
    """
    Rotate coordinates around a given axis.
    
    Parameters:
    -----------
    coords : np.ndarray
        Nx3 array of coordinates
    axis : str
        Axis to rotate around ('x', 'y', or 'z')
    angle_deg : float
        Rotation angle in degrees
    center : np.ndarray, optional
        Center point for rotation (if None, rotates around origin)
    
    Returns:
    --------
    rotated_coords : np.ndarray
        Rotated Nx3 coordinates
    """
    R = get_rotation_matrix(axis, angle_deg)
    
    if center is not None:
        # Translate to origin, rotate, translate back
        coords_centered = coords - center
        rotated = np.dot(coords_centered, R.T)
        return rotated + center
    else:
        return np.dot(coords, R.T)



def get_scale_bar_size(image_box_size):
    scale_bar_sizes_kpc = np.array([30, 20, 15, 10, 8, 5, 2, 1])
    compare_box_size_max = scale_bar_sizes_kpc*5
    compare_box_size_min = scale_bar_sizes_kpc*4

    try:
        multiplier = "kpc"
        for i in range(len(scale_bar_sizes_kpc)):
            #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
            if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_kpc[i]}")
                scale_bar_value = scale_bar_sizes_kpc[i]
                break
            if image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_kpc[i]}")
                scale_bar_value = scale_bar_sizes_kpc[i]
                break        

        print (f"Selected scale bar size: {scale_bar_value} kpc for image box size: {image_box_size} kpc")

    except:
        multiplier = "pc"
        try:
            scale_bar_sizes_pc = np.array([500, 300, 200, 100, 75, 50, 25, 20, 15, 10, 5, 2, 1, 0.5, 0.2, 0.1])
            compare_box_size_max = scale_bar_sizes_pc*5
            compare_box_size_min = scale_bar_sizes_pc*4

            #image_box_sizes = np.linspace(100, 4, 10)
            image_box_size = image_box_size*kpc/pc
            for i in range(len(scale_bar_sizes_pc)):
                #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
                if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                    #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_pc[i]}")
                    scale_bar_value = scale_bar_sizes_pc[i]
                    break
                elif image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                    #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_pc[i]}")
                    scale_bar_value = scale_bar_sizes_pc[i]
                    break        
                else:
                    scale_bar_value = scale_bar_sizes_pc[0]
                    #print (f"Image box size {image_box_size} not in range for scale bar size {scale_bar_sizes_pc[i]}")
                #    scale_bar_value = scale_bar_sizes_pc[i]
                    #break

            print (f"Selected scale bar size: {scale_bar_value} pc for image box size: {image_box_size} pc")
        except:
            multiplier = "au"
            try:
                scale_bar_sizes_au = np.array([10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1])
                compare_box_size_max = scale_bar_sizes_au*5
                compare_box_size_min = scale_bar_sizes_au*4

                #image_box_sizes = np.linspace(100, 4, 10)
                
                image_box_size = image_box_size*pc/AU
                for i in range(len(scale_bar_sizes_au)):
                    #print (f"Comparing image box size {image_box_size} to range {compare_box_size_min[i]} to {compare_box_size_max[i]}")
                    if image_box_size <= compare_box_size_max[i] and image_box_size >= compare_box_size_min[i]:
                        #print (f"Image box size {image_box_size} in range for scale bar size {scale_bar_sizes_au[i]}")
                        scale_bar_value = scale_bar_sizes_au[i]
                        break
                    elif image_box_size <= compare_box_size_min[i] and image_box_size > compare_box_size_max[i+1]:
                        #print (f"Image box size {image_box_size} less than range for scale bar size {scale_bar_sizes_au[i]}")
                        scale_bar_value = scale_bar_sizes_au[i]
                        break        
                    else:
                        scale_bar_value = scale_bar_sizes_au[0]
                        #print (f"Image box size {image_box_size} not in range for scale bar size {scale_bar_sizes_au[i]}")
                    #    scale_bar_value = scale_bar_sizes_au[i]
                        #break

                print (f"Selected scale bar size: {scale_bar_value} au for image box size: {image_box_size} au")
            except:
                scale_bar_value = scale_bar_sizes_au[-1]
                print (f"Could not determine scale bar size automatically, using previous value of {scale_bar_value} AU.")        
                #scale_bar_value

    if multiplier == "pc":
        if scale_bar_value < 1:
            return scale_bar_value * pc / kpc, f"{scale_bar_value:.1f} pc"
        else:
            return scale_bar_value * pc / kpc, f"{int(scale_bar_value)} pc"
    elif multiplier == "au":
        return scale_bar_value * AU / kpc, f"{int(scale_bar_value)} AU"
    elif multiplier == "kpc":
        return scale_bar_value, f"{int(scale_bar_value)} kpc"
    else:
        return None





def plot_image(snap_num, com, pdata, stardata, image_box_size, gas_dists, axis='z', angle_deg=0, res=1024, fire_units=True, aspect="square", cmap='viridis', save_path='./'):
    """Create surface density plot centered on densest gas particle."""
    fig, ax = plt.subplots()
    if aspect=="rectangle":
        fig.set_size_inches(16, 9)
    else:
        fig.set_size_inches(10, 10)

    center = com
    dist_cut_off = image_box_size*2

    pos, mass, hsml = pdata["Coordinates"][gas_dists<dist_cut_off], pdata["Masses"][gas_dists<dist_cut_off], \
                                pdata["SmoothingLength"][gas_dists<dist_cut_off]

    new_coords = rotate_coordinates(pos, axis=axis, angle_deg=angle_deg, center=com)
    if len(stardata.keys()) > 0:
        new_star_coords = rotate_coordinates(stardata["Coordinates"], axis=axis, angle_deg=angle_deg, center=com)

    M = Meshoid(new_coords, mass, hsml)

    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y, indexing='ij')

    if fire_units:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m*1e10, center=center,
                                            size=image_box_size, res=res)/1e6
        # Set color scale limits
        #sigma_valid = sigma_gas_msun_pc2[sigma_gas_msun_pc2 > 0]
        #vmax = sigma_valid.max() if len(sigma_valid) > 0 else 1e3
        #vmin = vmax/1e3
        vmax = 5e7
        vmin = 1e4
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, 
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax), 
                         cmap=cmap)
    else:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m, center=center,
                                            size=image_box_size, res=res)
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, 
                         norm=colors.LogNorm(vmin=1, vmax=2e3), 
                         cmap='inferno')

    # Plot stars if present
    if len(stardata.keys()) > 0:
        ax.scatter(new_star_coords[:,0], new_star_coords[:,1], 
                  c='w', 
                  s=stardata["ProtoStellarRadius_inSolar"]*Rsun/kpc/image_box_size*100000, 
                  alpha=0.6, 
                  edgecolors='black', 
                  linewidths=0.5)

    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    plt.xticks([])
    plt.yticks([])




    scale_bar_size, scale_bar_text = get_scale_bar_size(image_box_size)

    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                            scale_bar_size, scale_bar_text, 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=scale_bar_size/100, 
                            fontproperties=fontprops)

    #
    #scalebar = AnchoredSizeBar(ax.transData,
    #                        image_box_size/4, f'{image_box_size/4:.1f} pc', 'upper left', 
    #                        pad=1,
    #                        color='white',
    #                        frameon=False,
    #                        size_vertical=image_box_size/100,
    #                        fontproperties=fontprops)

    ax.add_artist(scalebar)
    fig.set_facecolor('black') 


    plt.tight_layout()
    image_save_path = os.path.join(save_path, 'movie_frames')
    os.makedirs(image_save_path, exist_ok=True)
    
    save_file = f"frame_{snap_num:04d}.png"
    plt.savefig(os.path.join(image_save_path, save_file), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {save_file}")
    return os.path.join(image_save_path, save_file)




def find_densest_gas_center(pdata):
    """Find the position of the densest gas particle."""
    densities = pdata["Density"]
    densest_idx = np.argmax(densities)
    return pdata["Coordinates"][densest_idx]


def process_snapshot(snap_num, path, sim, snapshot_suffix, snapdir, refinement_tag, full_tag, 
                    image_box_size, axis, angle_deg, res, aspect, cmap, save_path, movie_tag):
    """Process a single snapshot and create image."""
    try:
        # Load snapshot data
        header, pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
            sim, path, snap_num, snapshot_suffix=snapshot_suffix,
            snapdir=snapdir, refinement_tag=refinement_tag, full_tag=full_tag, movie_tag=movie_tag)

        # Convert to physical units
        header, pdata, stardata, fire_stardata = convert_units_to_physical(header, pdata, stardata, fire_stardata)
        
        # Find center (densest gas particle)
        com = find_densest_gas_center(pdata)
        
        # Calculate distances from center
        gas_pos = pdata['Coordinates'] - com
        gas_dists = np.linalg.norm(gas_pos, axis=1)
        
        # Create plot
        plot_image(snap_num, com, pdata, stardata, image_box_size, gas_dists, 
                  axis=axis, angle_deg=angle_deg, res=res, fire_units=True, 
                  aspect=aspect, cmap=cmap, save_path=save_path)
        
        return snap_num, True
    except Exception as e:
        print(f"Error processing snapshot {snap_num}: {str(e)}")
        return snap_num, False


def plot_wrapper(args):
    """Wrapper function for parallel processing."""
    snap_num, config = args
    return process_snapshot(snap_num, **config)


if __name__ == '__main__':
    # ==================== CONFIGURATION ====================
    save_path = "/mnt/home/skhullar/projects/SFIRE/m12f/new_jeans_refinement_movie/"
    
    # Simulation parameters
    path = "/mnt/home/skhullar/ceph/projects/SFIRE/m12f/"
    sim = "output_new_jeans_refinement"
    snapshot_suffix = ""
    start_snap = 28
    end_snap = 500  # Adjust as needed
    snapdir = False
    refinement_tag = False
    full_tag = False
    movie_tag= True
    
    # Image parameters
    image_box_size = 1e-5 #10.0  # kpc - adjust to zoom level desired
    axis = 'z'  # 'x', 'y', or 'z'
    angle_deg = 0  # rotation angle
    res = 1024  # resolution
    aspect = "square"  # "square" or "rectangle"
    cmap = 'cet_fire'  # colormap
    
    # Parallelization settings
    parallelize = True
    num_cores = 16
    
    # ========================================================
    
    print("="*70)
    print("Creating movie frames centered on densest gas particle")
    print("="*70)
    print(f"Simulation: {sim}")
    print(f"Snapshots: {start_snap} to {end_snap}")
    print(f"Box size: {image_box_size} kpc")
    print(f"Resolution: {res}x{res}")
    print(f"Output: {save_path}")
    print("="*70)
    
    # Prepare configuration dict
    config = {
        'path': path,
        'sim': sim,
        'snapshot_suffix': snapshot_suffix,
        'snapdir': snapdir,
        'refinement_tag': refinement_tag,
        'full_tag': full_tag,
        'movie_tag': movie_tag,
        'image_box_size': image_box_size,
        'axis': axis,
        'angle_deg': angle_deg,
        'res': res,
        'aspect': aspect,
        'cmap': cmap,
        'save_path': save_path
        }
    
    snapshot_list = list(range(start_snap, end_snap))
    
    if parallelize:
        print(f"\nUsing {num_cores} cores for parallel processing...")
        print(f"Total snapshots to process: {len(snapshot_list)}\n")
        
        # Create arguments list
        args_list = [(snap, config) for snap in snapshot_list]
        
        # Process in parallel
        start_time = time.time()
        with Pool(num_cores, maxtasksperchild=10) as pool:
            results = pool.map(plot_wrapper, args_list)
        
        total_time = time.time() - start_time
        successful = sum(1 for _, success in results if success)
        print(f"\nProcessed {successful}/{len(snapshot_list)} snapshots successfully")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    else:
        print(f"\nProcessing {len(snapshot_list)} snapshots sequentially...\n")
        start_time = time.time()
        successful = 0
        
        for snap_num in snapshot_list:
            _, success = process_snapshot(snap_num, **config)
            if success:
                successful += 1
            
            if (snap_num - start_snap + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (snap_num - start_snap + 1)
                remaining = len(snapshot_list) - (snap_num - start_snap + 1)
                est_remaining = avg_time * remaining
                print(f"Progress: {snap_num - start_snap + 1}/{len(snapshot_list)} | "
                      f"Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_remaining/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"\nProcessed {successful}/{len(snapshot_list)} snapshots successfully")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    print("\n" + "="*70)
    print("Done! Frames saved to:", os.path.join(save_path, 'movie_frames'))
    print("="*70)





