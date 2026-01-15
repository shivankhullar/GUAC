#!/usr/bin/env python3

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
from meshoid import GridSurfaceDensity

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
from scipy.interpolate import interp1d, UnivariateSpline


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





def collect_colorbar_ranges(count, com, pdata, stardata, image_box_size, gas_dists, axis, angle_deg, res=1024):
    """Collect min/max values for colorbars without saving images."""
    center = com
    dist_cut_off = image_box_size*2

    pos, masses, hsml = pdata["Coordinates"][gas_dists<dist_cut_off], pdata["Masses"][gas_dists<dist_cut_off], \
                                    pdata["SmoothingLength"][gas_dists<dist_cut_off]
    vx = pdata["Velocities"][:, 0][gas_dists<dist_cut_off]
    vy = pdata["Velocities"][:, 1][gas_dists<dist_cut_off]
    vz = pdata["Velocities"][:, 2][gas_dists<dist_cut_off]
    temps = pdata["Temperature"][gas_dists<dist_cut_off]

    new_coords = rotate_coordinates(pos, axis=axis, angle_deg=angle_deg, center=com)

    sigma_gas = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)
    temp_gas = GridSurfaceDensity(masses*temps, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    sigma_1Dz = GridSurfaceDensity(masses * vz**2, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    v_avg = GridSurfaceDensity(masses * vz, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    sigma_1D_final = np.sqrt(sigma_1Dz - v_avg**2) / 1e3

    res_16 = res/16
    cropped_sigma = sigma_gas[:, int(3.5*res_16):int(12.5*res_16)]
    cropped_sigma_1D = sigma_1D_final[:, int(3.5*res_16):int(12.5*res_16)]
    cropped_temp = temp_gas[:, int(3.5*res_16):int(12.5*res_16)]
    
    # Filter out zeros and invalid values for proper min calculation
    sigma_valid = cropped_sigma[cropped_sigma > 0]
    sigma_1D_valid = cropped_sigma_1D[cropped_sigma_1D > 0]
    temp_valid = cropped_temp[cropped_temp > 0]
    
    ranges = {
        'sigma_gas_min': sigma_valid.min() if len(sigma_valid) > 0 else 1e-10,
        'sigma_gas_max': sigma_valid.max() if len(sigma_valid) > 0 else 1e10,
        'sigma_1D_min': sigma_1D_valid.min() if len(sigma_1D_valid) > 0 else 1e-10,
        'sigma_1D_max': sigma_1D_valid.max() if len(sigma_1D_valid) > 0 else 1e10,
        'temp_gas_min': temp_valid.min() if len(temp_valid) > 0 else 1e1,
        'temp_gas_max': temp_valid.max() if len(temp_valid) > 0 else 1e8
    }
    
    print(f"Frame {count:04d}: box_size={image_box_size:.2e} kpc | sigma_gas=[{ranges['sigma_gas_min']:.2e}, {ranges['sigma_gas_max']:.2e}] | sigma_1D=[{ranges['sigma_1D_min']:.2e}, {ranges['sigma_1D_max']:.2e}] | temp=[{ranges['temp_gas_min']:.2e}, {ranges['temp_gas_max']:.2e}]")
    
    return ranges


def plot_image(count, com, pdata, stardata, image_box_size, gas_dists, axis, angle_deg, res=1024, fire_units=True, aspect="rectangle", cmap='cividis', save_path='./', colorbar_limits=None):
    #fire_units = True
    #res = 800
    fig, ax = plt.subplots()
    if aspect=="rectangle":
        fig.set_size_inches(16, 9)
        #res = 1920
    else:
        fig.set_size_inches(8,8)

    res_16 = res/16
    center = com
    dist_cut_off = image_box_size*2    
    #dist_cut_off = image_box_size*4

    pos, masses, hsml = pdata["Coordinates"][gas_dists<dist_cut_off], pdata["Masses"][gas_dists<dist_cut_off], \
                                    pdata["SmoothingLength"][gas_dists<dist_cut_off]
    vx = pdata["Velocities"][:, 0][gas_dists<dist_cut_off]
    vy = pdata["Velocities"][:, 1][gas_dists<dist_cut_off]
    vz = pdata["Velocities"][:, 2][gas_dists<dist_cut_off]
    temps = pdata["Temperature"][gas_dists<dist_cut_off]




    #new_coords = rotate_coordinates(pdata["Coordinates"], axis='z', angle_deg=0, center=com)
    #new_star_coords = rotate_coordinates(stardata["Coordinates"], axis='z', angle_deg=0, center=com)

    new_coords = rotate_coordinates(pos, axis=axis, angle_deg=angle_deg, center=com)
    new_star_coords = rotate_coordinates(stardata["Coordinates"], axis=axis, angle_deg=angle_deg, center=com)



    new_star_coords_pixels = (new_star_coords - (com - image_box_size/2)) * res / image_box_size
    new_star_coords_pixels_cropped = new_star_coords_pixels.copy()
    new_star_coords_pixels_cropped[:, 1] -= 3.5 * res_16

    if len(stardata.keys()) > 0:
        ax.scatter(new_star_coords_pixels_cropped[:,0], new_star_coords_pixels_cropped[:,1], 
                c='w', s=stardata["ProtoStellarRadius_inSolar"]*Rsun/kpc/image_box_size*2e5, alpha=0.6)




    sigma_gas = GridSurfaceDensity(masses, new_coords, hsml, com, image_box_size, res=res)
    #sigma_gas[sigma_gas==0] = 1e-3
    temp_gas = GridSurfaceDensity(masses*temps, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    #temp_gas[temp_gas==0] = 1e5
    #sigma_1Dx = GridSurfaceDensity(masses * vx**2, new_coords, hsml, com, image_box_size, res=1024)/sigma_gas
    #sigma_1Dy = GridSurfaceDensity(masses * vy**2, new_coords, hsml, com, image_box_size, res=1024)/sigma_gas
    sigma_1Dz = GridSurfaceDensity(masses * vz**2, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    v_avg = GridSurfaceDensity(masses * vz, new_coords, hsml, com, image_box_size, res=res)/sigma_gas
    sigma_1D_final = np.sqrt(sigma_1Dz - v_avg**2) / 1e3


    
    
    # Use provided colorbar limits or compute automatically
    if colorbar_limits is not None:
        sigma_vmin = colorbar_limits['sigma_gas_min']
        sigma_vmax = colorbar_limits['sigma_gas_max']
        sigma_1D_vmin = colorbar_limits['sigma_1D_min']
        sigma_1D_vmax = colorbar_limits['sigma_1D_max']
        temp_vmin = colorbar_limits['temp_gas_min']
        temp_vmax = colorbar_limits['temp_gas_max']
    else:
        cropped_sigma = sigma_gas[:, int(3.5*res_16):int(12.5*res_16)]
        cropped_sigma_1D = sigma_1D_final[:, int(3.5*res_16):int(12.5*res_16)]
        cropped_temp = temp_gas[:, int(3.5*res_16):int(12.5*res_16)]
        sigma_valid = cropped_sigma[cropped_sigma > 0]
        sigma_1D_valid = cropped_sigma_1D[cropped_sigma_1D > 0]
        temp_valid = cropped_temp[cropped_temp > 0]
        sigma_vmin = sigma_valid.min() if len(sigma_valid) > 0 else 1e-10
        sigma_vmax = sigma_valid.max() if len(sigma_valid) > 0 else 1e10
        sigma_1D_vmin = sigma_1D_valid.min() if len(sigma_1D_valid) > 0 else 1e-10
        sigma_1D_vmax = sigma_1D_valid.max() if len(sigma_1D_valid) > 0 else 1e10
        temp_vmin = temp_valid.min() if len(temp_valid) > 0 else 1e1
        temp_vmax = temp_valid.max() if len(temp_valid) > 0 else 1e8
    
    im = ax.imshow(temp_gas[:, int(3.5*res_16):int(12.5*res_16)].T, origin='lower', norm=LogNorm(vmin=temp_vmin, vmax=temp_vmax), cmap='cet_fire_r')
    im = ax.imshow(sigma_gas[:, int(3.5*res_16):int(12.5*res_16)].T, origin='lower', norm=LogNorm(vmin=sigma_vmin, vmax=sigma_vmax), cmap='cet_fire', alpha=0.8)
    im = ax.imshow(sigma_1D_final[:, int(3.5*res_16):int(12.5*res_16)].T, origin='lower', norm=LogNorm(vmin=sigma_1D_vmin, vmax=sigma_1D_vmax), cmap='cividis', alpha=0.5)

    

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
                            size_vertical=scale_bar_size_pixels/100, 
                            fontproperties=fontprops)


    ax.add_artist(scalebar)
    fig.set_facecolor('black') 

    plt.tight_layout()
    image_save_path = save_path + 'surf_dens/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    
    #image_box_size_pc = int(image_box_size*1e3)
    save_file = f"snap_{count:04d}.png"
    plt.savefig(image_save_path+save_file, dpi=100)#, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Explicitly close the figure
    print (f"Saved {save_file} to {save_path}....")


def save_colorbar_ranges(ranges_dict, save_path):
    """Save colorbar ranges to a text file."""
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')
    with open(ranges_file, 'w') as f:
        f.write("# Frame sigma_gas_min sigma_gas_max sigma_1D_min sigma_1D_max temp_gas_min temp_gas_max\n")
        for frame in sorted(ranges_dict.keys()):
            r = ranges_dict[frame]
            f.write(f"{frame} {r['sigma_gas_min']:.10e} {r['sigma_gas_max']:.10e} {r['sigma_1D_min']:.10e} {r['sigma_1D_max']:.10e} {r['temp_gas_min']:.10e} {r['temp_gas_max']:.10e}\n")
    print(f"Saved colorbar ranges to {ranges_file}")


def load_and_interpolate_colorbar_ranges(save_path, num_frames):
    """Load colorbar ranges and interpolate smoothly across all frames."""
    ranges_file = os.path.join(save_path, 'colorbar_ranges.txt')
    
    if not os.path.exists(ranges_file):
        print(f"Error: {ranges_file} not found. Run in 'collect' mode first.")
        return None
    
    # Load data
    data = np.loadtxt(ranges_file)
    frames = data[:, 0].astype(int)
    sigma_gas_min = data[:, 1]
    sigma_gas_max = data[:, 2]
    sigma_1D_min = data[:, 3]
    sigma_1D_max = data[:, 4]
    temp_gas_min = data[:, 5]
    temp_gas_max = data[:, 6]
    
    # Create smoothing spline functions (use log space for smooth transitions)
    # Using UnivariateSpline with s (smoothing factor) > 0 for smoothing
    # s=None means no smoothing (interpolation), s>0 means smoothing
    smoothing_factor = len(frames) * 0.004  # Adjust this for more/less smoothing
    interp_sigma_min = UnivariateSpline(frames, np.log10(sigma_gas_min), k=3, s=smoothing_factor)
    interp_sigma_max = UnivariateSpline(frames, np.log10(sigma_gas_max), k=3, s=smoothing_factor)
    
    smoothing_factor = len(frames) * 0.001  # Adjust this for more/less smoothing
    interp_sigma_1D_min = UnivariateSpline(frames, np.log10(sigma_1D_min), k=3, s=smoothing_factor)
    interp_sigma_1D_max = UnivariateSpline(frames, np.log10(sigma_1D_max), k=3, s=smoothing_factor)
    interp_temp_min = UnivariateSpline(frames, np.log10(temp_gas_min), k=3, s=smoothing_factor)
    interp_temp_max = UnivariateSpline(frames, np.log10(temp_gas_max), k=3, s=smoothing_factor)
    
    # Generate interpolated values for all frames
    all_frames = np.arange(num_frames)
    interpolated_ranges = {}
    
    for frame in all_frames:
        interpolated_ranges[frame] = {
            'sigma_gas_min': 10**interp_sigma_min(frame),
            'sigma_gas_max': 10**interp_sigma_max(frame),
            'sigma_1D_min': 10**interp_sigma_1D_min(frame),
            'sigma_1D_max': 10**interp_sigma_1D_max(frame),
            'temp_gas_min': 10**interp_temp_min(frame),
            'temp_gas_max': 10**interp_temp_max(frame)
        }
    
    print(f"Loaded and smoothed colorbar ranges for {num_frames} frames using cubic splines")
    return interpolated_ranges




save_path = "/mnt/home/skhullar/projects/SFIRE/m12f/jeans_refinement_movie/"


path = "/mnt/home/skhullar/ceph/projects/SFIRE/m12f/"
sim = "output_jeans_refinement"
snapshot_suffix = ""
snap_num = 32
snapdir=False
#refinement_tag=True
refinement_tag=False
full_tag=False
movie_tag=True

print ("Loading data...")

pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
        sim, path, snap_num, snapshot_suffix=snapshot_suffix, snapdir=snapdir, refinement_tag=refinement_tag, full_tag=full_tag, movie_tag=movie_tag)

print ("Loaded data...")

pdata, stardata, fire_stardata = convert_units_to_physical(pdata, stardata, fire_stardata)




com = np.median(stardata["Coordinates"], axis=0)

gas_pos = pdata['Coordinates'] - com
gas_dists = np.linalg.norm(gas_pos, axis=1)




#image_box_sizes1 = np.logspace(np.log10(120), np.log10(1e-3), 450)
#image_box_sizes2 = np.logspace(np.log10(1e-3), np.log10(1e-6), 150)
#image_box_sizes3 = np.logspace(np.log10(1e-6), np.log10(2e-5), 100)
#image_box_sizes_temp1 = np.append(image_box_sizes1, image_box_sizes2) 
#image_box_sizes_temp2 = np.repeat(2e-5,200)
#image_box_sizes = np.append(image_box_sizes_temp1, image_box_sizes_temp2)

image_box_sizes= np.append(np.logspace(np.log10(135), np.log10(1e-3), 450), np.append(np.logspace(np.log10(1e-3), np.log10(5e-6), 250), np.append(np.logspace(np.log10(5e-6), np.log10(2e-5), 200), np.repeat(2e-5,100))))
axes = ["y"]*600+["z"]*200+["y"]*200
angle_degs = np.append(np.linspace(90,360,300), np.append(np.linspace(0,90,100), np.append(np.repeat(90, 100), np.append(np.linspace(90,0,100), np.append(np.repeat(0, 200), np.linspace(0, 90, 200))))))


#print ("Box sizes: ", image_box_sizes, len(image_box_sizes))

#angle_deg = np.linspace(0,360,200)
#["x" for i in range(0, len(angle_deg))] + ["y" for i in range(0, len(angle_deg))] + ["y" for i in range(0, len(angle_deg))] + ["x" for i in range(0, len(angle_deg))]



#axes3 = ["x" for i in range(0, len(angle_deg))]
#axes4 = [""]
#new_coords = rotate_coordinates(pdata["Coordinates"], axis='z', angle_deg=0, center=com)
#new_star_coords = rotate_coordinates(stardata["Coordinates"], axis='z', angle_deg=0, center=com)

# Mode settings
# 'collect': First pass - collect colorbar ranges and save to file
# 'plot': Second pass - load ranges, interpolate, and create images with smooth colorbars
MODE = 'plot'  # Change to 'plot' for second pass, 'collect' for first pass

# Parallelization settings
parallelize = True
num_cores = 24  # Reduce cores - too many can cause overhead

# Global timing variables
start_time = time.time()
completed_count = 0
timing_lock = None
colorbar_ranges = {}  # For collecting ranges in 'collect' mode
interpolated_ranges = None  # For storing interpolated ranges in 'plot' mode

def collect_wrapper(args):
    """Wrapper function for collecting colorbar ranges"""
    global completed_count, start_time, colorbar_ranges
    count, image_box_size = args
    ranges = collect_colorbar_ranges(count, com, pdata, stardata, image_box_size, gas_dists, axes[count], angle_degs[count], res=1920)
    colorbar_ranges[count] = ranges
    
    # Update progress
    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / completed_count
        remaining = len(image_box_sizes) - completed_count
        est_remaining = avg_time * remaining
        print(f"Collecting Progress: {completed_count}/{len(image_box_sizes)} | Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_remaining/60:.1f}min")
    
    return count, ranges


def plot_wrapper(args):
    """Wrapper function for parallel processing with colorbar limits"""
    global completed_count, start_time, interpolated_ranges
    count, image_box_size = args
    colorbar_limits = interpolated_ranges[count] if interpolated_ranges is not None else None
    plot_image(count, com, pdata, stardata, image_box_size, gas_dists, axes[count], angle_degs[count], res=1920, fire_units=True, aspect="rectangle", cmap='cividis', save_path=save_path, colorbar_limits=colorbar_limits)
    
    # Update progress
    completed_count += 1
    if completed_count % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / completed_count
        remaining = len(image_box_sizes) - completed_count
        est_remaining = avg_time * remaining
        print(f"Plotting Progress: {completed_count}/{len(image_box_sizes)} | Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_remaining/60:.1f}min")

count=0

if MODE == 'collect':
    print("=" * 70)
    print("MODE: COLLECT - Gathering colorbar ranges from all frames")
    print("=" * 70)
    
    if parallelize:
        print(f"Using {num_cores} cores for parallel processing...")
        print(f"Total frames to process: {len(image_box_sizes)}")
        
        # Create arguments list: (count, image_box_size) pairs
        args_list = [(i, box_size) for i, box_size in enumerate(image_box_sizes)]
        
        # Run in parallel with maxtasksperchild to avoid memory issues
        start_time = time.time()
        with Pool(num_cores, maxtasksperchild=50) as pool:
            results = pool.map(collect_wrapper, args_list)
        
        # Collect results
        for count, ranges in results:
            colorbar_ranges[count] = ranges
        
        total_time = time.time() - start_time
        print(f"Collection time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    else:
        print(f"Total frames to process: {len(image_box_sizes)}")
        start_time = time.time()
        for count, image_box_size in enumerate(image_box_sizes):
            ranges = collect_colorbar_ranges(count, com, pdata, stardata, image_box_size, gas_dists, axes[count], angle_degs[count], res=1920)
            colorbar_ranges[count] = ranges
            
            if (count + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (count + 1)
                remaining = len(image_box_sizes) - (count + 1)
                est_remaining = avg_time * remaining
                print(f"Collecting Progress: {count+1}/{len(image_box_sizes)} | Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_remaining/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"Collection time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    # Save collected ranges
    save_colorbar_ranges(colorbar_ranges, save_path)
    print("\nColorbar ranges collected and saved!")
    print("Next step: Change MODE to 'plot' and run again to generate images with smooth colorbars.")

elif MODE == 'plot':
    print("=" * 70)
    print("MODE: PLOT - Creating images with smoothly interpolated colorbars")
    print("=" * 70)
    
    # Load and interpolate colorbar ranges
    interpolated_ranges = load_and_interpolate_colorbar_ranges(save_path, len(image_box_sizes))
    
    if interpolated_ranges is None:
        print("Error: Could not load colorbar ranges. Exiting.")
    else:
        if parallelize:
            print(f"Using {num_cores} cores for parallel processing...")
            print(f"Total frames to process: {len(image_box_sizes)}")
            
            # Create arguments list: (count, image_box_size) pairs
            args_list = [(i, box_size) for i, box_size in enumerate(image_box_sizes)]
            
            # Run in parallel with maxtasksperchild to avoid memory issues
            start_time = time.time()
            with Pool(num_cores, maxtasksperchild=50) as pool:
                pool.map(plot_wrapper, args_list)
            
            total_time = time.time() - start_time
            print(f"Total plotting time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        else:
            print(f"Total frames to process: {len(image_box_sizes)}")
            start_time = time.time()
            for count, image_box_size in enumerate(image_box_sizes):
                colorbar_limits = interpolated_ranges[count]
                plot_image(count, com, pdata, stardata, image_box_size, gas_dists, axes[count], angle_degs[count], res=1920, fire_units=True, aspect="rectangle", cmap='cividis', save_path=save_path, colorbar_limits=colorbar_limits)
                
                if (count + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (count + 1)
                    remaining = len(image_box_sizes) - (count + 1)
                    est_remaining = avg_time * remaining
                    print(f"Plotting Progress: {count+1}/{len(image_box_sizes)} | Elapsed: {elapsed/60:.1f}min | Est. remaining: {est_remaining/60:.1f}min")
            
            total_time = time.time() - start_time
            print(f"Total plotting time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

else:
    print(f"Error: Invalid MODE '{MODE}'. Must be 'collect' or 'plot'.")

print ("\n" + "="*70)
print ("Done!")
print ("="*70)






