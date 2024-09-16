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
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --threshold=<threshold>                             Threshold for cloud selection [default: 0.3]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: m12i_final_fb_7k]
    --save_path=<save_path>                             Path to save the images [default: ./]
    --image_box_size=<image_box_size>                   Size of the image box [default: 0.5]
    --snapnum_range=<snapnum_range>                     Range of snapshots to plot [default: 0,100]
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *

from docopt import docopt
from meshoid import Meshoid
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import os



def get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=''):

    if snap<10:
        snapname = 'snapshot_'+snapshot_suffix+'00{num}'.format(num=snap) 
    elif snap>=10 and snap<100:
        snapname = 'snapshot_'+snapshot_suffix+'0{num}'.format(num=snap)
    else:
        snapname = 'snapshot_'+snapshot_suffix+'{num}'.format(num=snap) 

    filename = sim_path+sim+'/snapshots/'+snapname+'.hdf5'        
    print ("Reading file:", filename)
    F = h5py.File(filename,"r")
    pdata = {}
    for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber":
        pdata[field] = F["PartType0"][field][:]#[density_cut]

    try:
        pdata['RefinementFlag'] = F["PartType0"]['RefinementFlag'][:]
    except:
        pass

    for key in F['Header'].attrs.keys():
        pdata[key] = F['Header'].attrs[key]
    
    stardata = {}
    if 'PartType5' in F.keys():
        try:
            for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber", "StellarFormationTime":
                stardata[field] = F["PartType5"][field][:]#[density_cut]

            for key in F['Header'].attrs.keys():
                stardata[key] = F['Header'].attrs[key]
        except:
            print('No STARFORGE stars data in this snapshot')

    fire_stardata = {}
    if 'PartType4' in F.keys():
        try:
            for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber":
                fire_stardata[field] = F["PartType4"][field][:]#[density_cut]

            for key in F['Header'].attrs.keys():
                fire_stardata[key] = F['Header'].attrs[key]
        except:
            print('No FIRE stars data in this snapshot')
    refine_data = {}
    if 'PartType3' in F.keys():
        for field in "Masses", "Coordinates", "Velocities":
            refine_data[field] = F["PartType3"][field][:]#[density_cut]

    #refine_pos = np.array(F['PartType3/Coordinates'])
    #refine_pos = refine_pos[0]
    F.close()

    return pdata, stardata, fire_stardata, refine_data, snapname


def plot_dist_res(pdata, snap_num, gas_dists, save_path):

    
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    plt.scatter(gas_dists[gas_dists<2], pdata['Masses'][gas_dists<2]*1e10, s=1, alpha=0.5)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from refine center (cosmo units) [kpc]', fontsize=18)
    ax.set_ylabel('Mass (cosmo units) [M$_{\\odot}$]', fontsize=18)

    #ax.legend(loc=5, prop={'size': 16})

    ax.set_xlim([1e-3, 2])
    ax.set_ylim([0.1, 2e4])
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


def plot_surf_dens(image_box_size, pdata, snap_num, com, gas_dists, save_path):
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
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m*1e10,center=center,\
                                            size=image_box_size,res=res)/1e6 #*1e4
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1e-2,vmax=1e4), cmap='inferno')
    #ax.scatter(tagged_coords[:,0], tagged_coords[:,1], c='r', s=1, alpha=0.5)
    ax.scatter(com[0], com[1], c='g', s=10)
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


if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    save_path = path+sim+'/'+args['--save_path']
    image_box_size = float(args['--image_box_size'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    print (save_path)
    #sim = 'gas_pID_test'
    
    #sim = 'orig_tag_test'
    #path = '/mnt/raid-project/murray/khullar/GMC_IC_Project/Sims/refinetag_test/'
    
    for snap_num in range(snapnum_range[0], snapnum_range[1]+1):
        print ("==============================================")
        print ("Snapshot number:", snap_num)
        snapshot_suffix = ''
        pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, path, \
                                                                    snap_num, snapshot_suffix=snapshot_suffix)

        inds = np.where(pdata["RefinementFlag"]==1)[0] 
        tagged_coords = pdata["Coordinates"][inds]
        tagged_masses = pdata["Masses"][inds]

        com_x = np.sum(tagged_coords[:,0]*1/tagged_masses)/np.sum(1/tagged_masses)
        com_y = np.sum(tagged_coords[:,1]*1/tagged_masses)/np.sum(1/tagged_masses)
        com_z = np.sum(tagged_coords[:,2]*1/tagged_masses)/np.sum(1/tagged_masses)

        com = np.array([com_x, com_y, com_z])
        gas_pos = pdata['Coordinates'] - com
        gas_dists = np.linalg.norm(gas_pos, axis=1)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        plot_dist_res(pdata, snap_num, gas_dists, save_path)

        print ("Done with plotting distance vs mass")
        plot_surf_dens(image_box_size, pdata, snap_num, com, gas_dists, save_path)

        print(f'Finished plotting for snapshot {snap_num}')
