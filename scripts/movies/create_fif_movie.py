#!/usr/bin/env python3

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from hybrid_sims_utils.read_snap import *

from meshoid import Meshoid
import matplotlib
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





def plot_image(com, new_coords, new_star_coords, pdata, stardata, image_box_size, res=1024, fire_units=True, aspect="rectangle", cmap='cividis'):
    #fire_units = True
    #res = 800
    fig, ax = plt.subplots()
    if aspect=="rectangle":
        fig.set_size_inches(16, 9)
        res = 1920
    else:
        fig.set_size_inches(8,8)

    center = com
    dist_cut_off = image_box_size*2

    pos, mass, hsml = new_coords[gas_dists<dist_cut_off], pdata["Masses"][gas_dists<dist_cut_off], \
                                pdata["SmoothingLength"][gas_dists<dist_cut_off]

    M = Meshoid(pos, mass, hsml)

    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    #box_size = max_pos-min_pos
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y, indexing='ij')

    if fire_units:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m*1e10,center=center,\
                                            size=image_box_size,res=res)/1e6 #*1e4
        if aspect == "rectangle":
            p = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=sigma_gas_msun_pc2.min(),vmax=sigma_gas_msun_pc2.max()), cmap=cmap)
        else:
            p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=sigma_gas_msun_pc2.min(),vmax=sigma_gas_msun_pc2.max()), cmap=cmap)      

    else:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1,vmax=2e3), cmap='inferno')


    if len(stardata.keys()) > 0:
        ax.scatter(new_star_coords[:,0], new_star_coords[:,1], c='w', s=stardata["ProtoStellarRadius_inSolar"]*Rsun/kpc/image_box_size*10000, alpha=0.5)

    #if len(fire_stardata.keys()) > 0:
    #    ax.scatter(fire_stardata['Coordinates'][:,0], fire_stardata['Coordinates'][:,1], c='g', s=10, alpha=0.5)
    if aspect == "rectangle":
        ax.set_xlim([min_pos[0], max_pos[0]])
        ax.set_ylim([min_pos[1]+(3.5*image_box_size/16), max_pos[1]-(3.5*image_box_size/16)])
    else:
        ax.set_aspect('equal')
        ax.set_xlim([min_pos[0], max_pos[0]])
        ax.set_ylim([min_pos[1], max_pos[1]])
    #plt.colorbar(p, label='Surface Density [M$_\odot$/pc$^2$]')
    plt.xticks([])
    plt.yticks([])




    scale_bar_size, scale_bar_unit = get_scale_bar_size(image_box_size)

    if scale_bar_size < 1:
        scale_bar_text = f"{scale_bar_size:.1f}"+scale_bar_unit
    else:
        scale_bar_text = f"{scale_bar_size}"+scale_bar_unit

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

    plt.tight_layout()
    #image_save_path = save_path + 'surf_dens/'
    #if not os.path.exists(image_save_path):
    #    os.makedirs(image_save_path)

    #image_box_size_pc = int(image_box_size*1e3)









path = "/mnt/home/skhullar/ceph/projects/SFIRE/m12f/"
sim = "output_jeans_refinement"
snapshot_suffix = ""
snap_num = 32
snapdir=False
#refinement_tag=True
refinement_tag=False
full_tag=True

pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
        sim, path, snap_num, snapshot_suffix=snapshot_suffix, snapdir=snapdir, refinement_tag=refinement_tag, full_tag=full_tag)



pdata, stardata, fire_stardata = convert_units_to_physical(pdata, stardata, fire_stardata)




com = np.median(stardata["Coordinates"], axis=0)

gas_pos = pdata['Coordinates'] - com
gas_dists = np.linalg.norm(gas_pos, axis=1)



image_box_sizes1 = np.logspace(np.log10(120), np.log10(1e-3), 200)
image_box_sizes2 = np.logspace(np.log10(1e-3), np.log10(1e-5), 500)
image_box_sizes = np.append(image_box_sizes1[:-1], image_box_sizes2) 



for image_box_size in image_box_sizes:
    plot_image(new_coords, new_star_coords, image_box_size)






new_coords = rotate_coordinates(pdata["Coordinates"], axis='x', angle_deg=90, center=com)
new_star_coords = rotate_coordinates(stardata["Coordinates"], axis='x', angle_deg=90, center=com)


