#!/usr/bin/env python
"""
step_refine_movie.py: "Create initial conditions file for FIRE+STARFORGE simulation 
                            with a refinement particle placed at a custom position.
                            The custom position is specified by the user. The script will
                            then choose particles within a certain distance of the custom
                            position to calculate the center of mass velocity of the SMBH particle.
                            One can specify if the particles should follow stars or gas particles while
                            deciding the original position for the refinement particle.
                            The script will then create an hdf5 file with the ICs."

Usage: step_refine_movie.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir or snapshot directory? [default: True]
    --threshold=<threshold>                             Threshold for cloud selection [default: 0.3]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: m12i_final_fb_7k]
    --save_path=<save_path>                             Path to save the images [default: ./]
    --image_box_size=<image_box_size>                   Size of the image box [default: 0.5]
    --refinement_flag=<refinement_flag>                 Should the refinement flag be used? [default: False]
    --snapnum_range=<snapnum_range>                     Range of snapshots to plot [default: 0,100]
    --parallel=<parallel>                               Should the script execute in parallel? [default: False]
    --num_cores=<num_cores>                             Number of processors to run on [default: 4]
    --fire_units=<fire_units>                           Should the script use FIRE units? [default: True]
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
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import os




def plot_dist_res(pdata, snap_num, gas_dists, save_path, fire_units=True):

    
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from refine center (cosmo units) [kpc]', fontsize=18)
    ax.set_ylabel('Mass (cosmo units) [M$_{\\odot}$]', fontsize=18)

    #ax.legend(loc=5, prop={'size': 16})
    if fire_units:
        plt.scatter(gas_dists[gas_dists<2], pdata['Masses'][gas_dists<2]*1e10, s=1, alpha=0.5)
        ax.set_xlim([1e-3, 2])
        ax.set_ylim([0.1, 2e4])
        ax.set_xlabel('Distance from refine center (cosmo units) [kpc]', fontsize=18)
        ax.set_ylabel('Mass (cosmo units) [M$_{\\odot}$]', fontsize=18)

    else:
        plt.scatter(gas_dists, pdata['Masses'], s=1, alpha=0.5)
        ax.set_xlabel('Distance from refine center [pc]', fontsize=18)
        ax.set_ylabel('Mass [M$_{\\odot}$]', fontsize=18)



    ax.minorticks_on()
    ax.tick_params(which = 'both', direction = 'in', labelbottom=True, \
                                right = True, top=True, labelsize=16)
    ax.tick_params(which='major', length=9, width=1)
    ax.tick_params(which='minor', length=6, width=1)
    plt.tight_layout()

    image_save_path = save_path + 'mass_res_dist/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    plt.savefig(image_save_path+f'mass_res_dist_{snap_num}.png')
    plt.close()


def plot_surf_dens(image_box_size, pdata, snap_num, com, gas_dists, save_path, fire_units):
    res = 800
    #box_size = pdata["BoxSize"] #f['Header'].attrs['BoxSize'] #pdata['BoxSize']
    center = com
    dist_cut_off = image_box_size

    pos, mass, hsml = pdata["Coordinates"][gas_dists<dist_cut_off], pdata["Masses"][gas_dists<dist_cut_off], \
                                pdata["SmoothingLength"][gas_dists<dist_cut_off]
    #hsml = np.ones(len(mass))*1

    M = Meshoid(pos, mass, hsml)

    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    #box_size = max_pos-min_pos
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y, indexing='ij')
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    if fire_units:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m*1e10,center=center,\
                                            size=image_box_size,res=res)/1e6 #*1e4
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1e-2,vmax=5e4), cmap='inferno')
        ax.scatter(com[0], com[1], c='g', s=10)
    
    else:
        sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)
        p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1,vmax=2e3), cmap='inferno')
    
    #ax.scatter(tagged_coords[:,0], tagged_coords[:,1], c='r', s=1, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    #plt.colorbar(p, label='Surface Density [M$_\odot$/pc$^2$]')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    image_save_path = save_path + 'surf_dens/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    
    image_box_size_pc = int(image_box_size*1e3)
    plt.savefig(image_save_path+f'surf_dens_{image_box_size_pc}_{snap_num}.png')
    plt.close()



def process_snapshot(args):
    snap_num, sim, path, snapshot_suffix, snapdir, save_path, image_box_size, refinement_tag, fire_units = args
    pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
        sim, path, snap_num, snapshot_suffix=snapshot_suffix, snapdir=snapdir, refinement_tag=refinement_tag)

    if refinement_tag:
        inds = np.where(pdata["RefinementFlag"] == 1)[0]
    #inds = np.where(pdata["RefinementFlag"] == 1)[0]
        tagged_coords = pdata["Coordinates"][inds]
        tagged_masses = pdata["Masses"][inds]

        com_x = np.sum(tagged_coords[:, 0] * 1 / tagged_masses) / np.sum(1 / tagged_masses)
        com_y = np.sum(tagged_coords[:, 1] * 1 / tagged_masses) / np.sum(1 / tagged_masses)
        com_z = np.sum(tagged_coords[:, 2] * 1 / tagged_masses) / np.sum(1 / tagged_masses)

        com = np.array([com_x, com_y, com_z])
    else:
        com = np.array([pdata['BoxSize']/2, pdata['BoxSize']/2, pdata['BoxSize']/2])
        
    gas_pos = pdata['Coordinates'] - com
    gas_dists = np.linalg.norm(gas_pos, axis=1)

    plot_dist_res(pdata, snap_num, gas_dists, save_path, fire_units=fire_units)
    plot_surf_dens(image_box_size, pdata, snap_num, com, gas_dists, save_path, fire_units=fire_units)

    print(f'Finished plotting for snapshot {snap_num}')




if __name__ == '__main__':
    args = docopt(__doc__)
    parallel = convert_to_bool(args['--parallel'])
    path = args['--path']
    sim = args['--sim']
    num_cores = int(args['--num_cores'])
    snapdir = convert_to_bool(args['--snapdir'])
    save_path = path+sim+'/'+args['--save_path']
    image_box_size = float(args['--image_box_size'])
    refinement_tag = convert_to_bool(args['--refinement_flag'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    fire_units = convert_to_bool(args['--fire_units'])
    print (save_path)
    #sim = 'gas_pID_test'
    
    #sim = 'orig_tag_test'
    #path = '/mnt/raid-project/murray/khullar/GMC_IC_Project/Sims/refinetag_test/'
    
    
    snap_args = [(snap_num, sim, path, '', snapdir, save_path, image_box_size, refinement_tag, fire_units) for snap_num in range(snapnum_range[0], snapnum_range[1] + 1)]

    if parallel:
        pool = multiprocessing.Pool(processes=num_cores)
        pool.map(process_snapshot, snap_args)
        pool.close()
        pool.join()
    else:
        for args in snap_args:
            process_snapshot(args)
