#!/usr/bin/env python
"""
create_zoomin_movie: "Create a movie of a zoom-in simulation of a cloud in a galaxy."

Usage: create_zoomin_movie.py [options]

Options:
    -h, --help                  Show this screen
    --snapdir=<snapdir>         Are snapshots in a snapdir directory? [default: True]
    --threshold=<threshold>     Threshold for cloud selection [default: 0.3]
    --path=<path>               Path to the simulation directory [default: ./]
    --snapnum=<snapnum>         Snapshot number [default: 625]
    --cloud_num=<cloud_num>     Cloud number [default: 40]
    --nmin=<nmin>               Minimum number of particles in a cloud [default: 10]
    --vir=<vir>                 Virial parameter [default: 5]
    --ic_path=<ic_path>         Path to save the IC file [default: ./] 
"""



from docopt import docopt
import sys
sys.path.insert(0, '../src/')
#from IC_funcs import *
from epsff_calc import *
from viz_utils import *
from cloud_selector_funcs import *

from fire_utils import *

from meshoid import Meshoid

from matplotlib.colors import LogNorm
from matplotlib import colors
#%matplotlib inline

import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from cloud_analysis import *
#matplotlib.use('Agg')








if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    snapdir = args['--snapdir']
    ic_path = args['--ic_path']
    
        

    ## Some bookkeeping
    #path = "../../../FIRE/m12i_final/"
    start_snap = 600
    last_snap = 650
    #field_names = ["names", "masses"]
    filename_prefix = "Linked_Clouds_"

    #No. of digits in the names of the clouds and the snapshots.
    cloud_num_digits = 4
    snapshot_num_digits = 4
    cloud_prefix = "Cloud"
    image_path = 'img_data/'
    image_filename_prefix = 'center_proj_'
    image_filename_suffix = '.hdf5'
    hdf5_file_prefix = 'Clouds_'
    age_cut = 1
    dat_file_header_size=8
    snapshot_prefix="Snap"
    star_data_sub_dir = "StarData/"
    cph_sub_dir="CloudPhinderData/"
    #cph_sub_dir='m12i_restart/'
    frac_thresh='thresh'+str(args['--threshold'])

    r_gal = 25
    h = 0.4 #0.4

    #save_path = './data/'

    nmin = 10
    vir = 5
    sim = "m12i_final_fb_7k"
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
    filename = path+sub_dir+filename_prefix+str(start_snap)+"_"+str(last_snap)+"names"+".txt"

    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)

    snap_num = int(args['--snapnum'])


    snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
    dm_masses = load_from_snapshot.load_from_snapshot('Masses', 1, snapdir, snap_num, units_to_physical=True)

    dm_pos = load_from_snapshot.load_from_snapshot('Coordinates', 1, snapdir, snap_num, units_to_physical=True)


    gas_masses = load_from_snapshot.load_from_snapshot('Masses', 0, snapdir, snap_num, units_to_physical=True)
    gas_pos = load_from_snapshot.load_from_snapshot('Coordinates', 0, snapdir, snap_num, units_to_physical=True)
    gas_hsmls = load_from_snapshot.load_from_snapshot('SmoothingLength', 0, snapdir, snap_num, units_to_physical=True)


    gal_center = Get_Galaxy_Centre(params, snap_num)
    proj_matrix = Get_Galaxy_Proj_Matrix(params, snap_num)
    gal_center_proj = np.matmul(proj_matrix, gal_center)

    dm_pos_proj = np.ones(dm_pos.shape)
    for i in range(dm_pos.shape[0]):
        dm_pos_proj[i] = np.matmul(proj_matrix, dm_pos[i])

    gas_pos_proj = np.ones(gas_pos.shape)
    for i in range(gas_pos.shape[0]):
        gas_pos_proj[i] = np.matmul(proj_matrix, gas_pos[i])



    # We will progressively zoom in to the box. 
    pos, mass = dm_pos_proj, dm_masses
    hsml = np.ones(len(mass))*0.3
    M_dm = Meshoid(pos, mass, hsml)

    pos, mass, hsml = gas_pos_proj, gas_masses, gas_hsmls
    M_gas = Meshoid(pos, mass, hsml)

    initial_dm_alpha=1
    initial_gas_alpha=0
    final_gas_alpha=1
    final_dm_alpha=0
    # We will smoothly transition between the two after getting to a box size of 500
    # The transition will happen over 50 frames


    # Deal with slowing down the zoom in once we reach certain box sizes. Just give a list of box_breaks
    #box_breaks = [1000,200]
    # Decide what percentage of the total time to be spent on the given box sizes
    #box_size_time_percentages = [0.3, 0.3, 0.4] #should sum to 1 and be 1 more in length than box_breaks
    
    # Make an array of the box_sizes
    times = np.arange(0, 600)

    initial_box_size = 4000
    final_box_size = 5

    box_breaks = [1000,300]
    #box_breaks = [800,100]
    box_size_time_percentages = [0.3, 0.3, 0.4] #should sum to 1 and be 1 more in length than box_breaks
    box_sizes = np.linspace(initial_box_size, box_breaks[0], int(box_size_time_percentages[0]*len(times)))
    for i in range(1, len(box_breaks)):
        #print (np.linspace(box_sizes[-1], box_breaks[i], int(box_size_time_percentages[i]*len(times))))
        box_sizes = np.append(box_sizes[:-1], np.linspace(box_sizes[-1], box_breaks[i], int(box_size_time_percentages[i]*len(times))))

    box_sizes = np.append(box_sizes[:-1], np.linspace(box_sizes[-1], final_box_size, int(box_size_time_percentages[-1]*len(times))))
    box_sizes = np.append(box_sizes, np.ones(len(times)-len(box_sizes))*final_box_size)

    print (box_sizes)




    res = 800 #change to 2048 later maybe

    center = gal_center_proj

    transition_times = 30
    gas_transition_times = 30
    count = 0
    gas_count = 0
    for i in range(0, len(times)):
        time = i
        current_box_size = box_sizes[i] #initial_box_size - time*(initial_box_size-final_box_size)/len(times)
        
        image_box_size = current_box_size
        min_pos = center-image_box_size/2
        max_pos = center+image_box_size/2

        if current_box_size < 800:
            current_dm_alpha = initial_dm_alpha - count*(initial_dm_alpha-final_dm_alpha)/transition_times
            current_gas_alpha = initial_gas_alpha + count*(final_gas_alpha-initial_gas_alpha)/transition_times
            count += 1
            if current_dm_alpha < 0:
                current_dm_alpha = 0
            if current_gas_alpha > 1:
                current_gas_alpha = 1
        else:
            current_dm_alpha = 1
            current_gas_alpha = 0

        print (time, current_box_size, current_dm_alpha, current_gas_alpha)

        X = np.linspace(min_pos[0], max_pos[0], res)
        Y = np.linspace(min_pos[1], max_pos[1], res)
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots()
        fig.set_size_inches(10,10)
        if current_dm_alpha>0:
            sigma_dm_msun_pc2 = M_dm.SurfaceDensity(M_dm.m*1e10,center=center,\
                                                    size=image_box_size,res=res) #*1e4
            sigma_dm_msun_pc2[sigma_dm_msun_pc2<1e3] = 1e3
            p2 = ax.pcolormesh(X, Y, sigma_dm_msun_pc2, norm=colors.LogNorm(vmin=1e3,vmax=1e9), cmap='inferno', alpha=current_dm_alpha)

        if current_gas_alpha > 0:
            sigma_gas_msun_pc2 = M_gas.SurfaceDensity(M_gas.m*1e10,center=center,\
                                                size=image_box_size,res=res)/1e6 #*1e4
            #sigma_gas_msun_pc2[sigma_gas_msun_pc2<1] = 1
        
            #
            if current_box_size<500:
                current_gas_alpha1 = initial_dm_alpha - gas_count*(initial_dm_alpha-final_dm_alpha)/gas_transition_times
                current_gas_alpha2 = initial_gas_alpha + gas_count*(final_gas_alpha-initial_gas_alpha)/gas_transition_times
                if current_gas_alpha1 < 0:
                    current_gas_alpha1 = 0
                if current_gas_alpha2 > 1:
                    current_gas_alpha2 = 1

                print (current_gas_alpha1, current_gas_alpha2, gas_count)
                gas_count += 1
                if current_gas_alpha1 < 0:
                    current_gas_alpha1 = 0
                if current_gas_alpha2 > 1:
                    current_gas_alpha2 = 1
                p2 = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha1)
                p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1e-1,vmax=1e3), cmap='inferno', alpha=current_gas_alpha2)
            else:
                p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha)
            
        ax.set_aspect('equal')
        ax.set_xlim([min_pos[0], max_pos[0]])
        ax.set_ylim([min_pos[1], max_pos[1]])
        ax.set_xticks([])
        ax.set_yticks([])


        scale_bar_size = int(image_box_size/3)//10*10
        fontprops = fm.FontProperties(size=20)
        if scale_bar_size>=1000:
            scale_bar_size=1000
            scale_text = '{scale} Mpc'.format(scale=scale_bar_size//1000)
        elif scale_bar_size>=10 and scale_bar_size<1000:
            scale_text = '{scale} kpc'.format(scale=scale_bar_size)
        elif scale_bar_size<10:
            scale_bar_size = int(image_box_size/3)
            scale_text = '{scale} kpc'.format(scale=scale_bar_size)

        scalebar = AnchoredSizeBar(ax.transData,
                                scale_bar_size, scale_text, 'upper left', 
                                pad=1,
                                color='white',
                                frameon=False,
                                size_vertical=scale_bar_size/100, 
                                fontproperties=fontprops)

        ax.add_artist(scalebar)

        #plt.colorbar(p, label='Surface Density [M$_\odot$/pc$^2$]')
        plt.savefig('../movies/zoom_final/zoom_{time}.png'.format(time=time))
        plt.close()
