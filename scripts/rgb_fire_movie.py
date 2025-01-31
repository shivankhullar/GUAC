#!/usr/bin/env python
"""
rgb_fire_movie: "Script to make a sequence of RGB density, temperature, pressure composite of a FIRE galaxy"

Usage: rgb_fire_movie.py [options]

Options:
    -h, --help                                                  Show this screen
    --path=<path>                                               Path to the simulation directory [default: ./]
    --sim=<sim>                                                 Simulation name [default: ]
    --snapdir=<snapdir>                                         Are snapshots in a snapdir directory? [default: False]
    --snapnum=<snapnum>                                         Snapshot number [default: 600]
    --all_snaps_in_dir=<all_snaps_in_dir>                       Are all snapshots in the same directory? [default: False]
    --r_gal=<r_gal>                                             Galaxy radius [default: 25]
    --h=<h>                                                     Scale height [default: 3] 
    --save_path=<save_path>                                     Path to save the images [default: ./]
    --box_size=<box_size>                                       Size of the image box [default: 35]
    --age_cut=<age_cut>                                         Age cut [default: 3]
    --stars=<stars>                                             Include stars? [default: False]
    --res=<res>                                                 Resolution of the image [default: 1024]
    --rectangle_mode=<rectangle_mode>                           Final image 16:9 aspect ratio? [default: True]
"""


from docopt import docopt
from generic_utils.fire_utils import *
from generic_utils.script_utils import *
from generic_utils.load_fire_snap import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from galaxy_utils.gal_utils import *

import glob
import yt
import h5py
from meshoid import Meshoid
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import colorcet as cc
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def plot_rgb_galaxy(gal_quants0, snapnum, image_box_size, save_path, rectangle_mode=False):
    image_box_size = 20
    res = 2048
    center = gal_quants0.gal_centre_proj
    print ("Center:", center)
    print ("Coordinates:", gal_quants0.data["Coordinates"])

    smooth_fac = 1
    print ("Creating meshoid object...")
    M = Meshoid(gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], gal_quants0.data["SmoothingLength"])
    print (gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], gal_quants0.data["SmoothingLength"])
    #proj_grid = M.Projection(gal_quants0.data["Density"], center=center, size=image_box_size, res=1024)

    print ("Getting the projection maps...")
    pres_proj_grid = M.ProjectedAverage(gal_quants0.data["Pressure"], center=center, size=image_box_size, res=res, smooth_fac=smooth_fac)
    dens_proj_grid = M.ProjectedAverage(gal_quants0.data["Density"], center=center, size=image_box_size, res=res, smooth_fac=smooth_fac)
    temp_proj_grid = M.ProjectedAverage(gal_quants0.data["Temperature"], center=center, size=image_box_size, res=res, smooth_fac=smooth_fac)
    print ("Making the RGB image...")

    print ("Density Max, min:", dens_proj_grid.max(), dens_proj_grid.min())
    print ("Temperature Max, min:", temp_proj_grid.max(), temp_proj_grid.min())
    print ("Pressure Max, min:", pres_proj_grid.max(), pres_proj_grid.min())

    # We'll make an RGB image from these 3 maps
    # Normalizing data to lie between 0 and 1
    dens_range = [1e-4,1]
    clipped_dens_proj_grid = dens_proj_grid 
    clipped_dens_proj_grid[clipped_dens_proj_grid<dens_range[0]] = dens_range[0]
    clipped_dens_proj_grid[clipped_dens_proj_grid>dens_range[1]] = dens_range[1]
    norm_dens_proj_grid = np.log10(dens_proj_grid) - np.log10(dens_range[0])
    norm_dens_proj_grid = norm_dens_proj_grid/norm_dens_proj_grid.max()

    temp_range = [2e3, 2e6]
    clipped_temp_proj_grid = temp_proj_grid
    clipped_temp_proj_grid[clipped_temp_proj_grid<temp_range[0]] = temp_range[0]
    clipped_temp_proj_grid[clipped_temp_proj_grid>temp_range[1]] = temp_range[1]
    norm_temp_proj_grid = np.log10(temp_proj_grid) - np.log10(temp_range[0])
    norm_temp_proj_grid = norm_temp_proj_grid/norm_temp_proj_grid.max()

    pres_range = [1e-2, 10]
    clipped_pres_proj_grid = pres_proj_grid
    clipped_pres_proj_grid[clipped_pres_proj_grid<pres_range[0]] = pres_range[0]
    clipped_pres_proj_grid[clipped_pres_proj_grid>pres_range[1]] = pres_range[1]
    norm_pres_proj_grid = np.log10(pres_proj_grid) - np.log10(pres_range[0])
    norm_pres_proj_grid = norm_pres_proj_grid/norm_pres_proj_grid.max()

    for combo in range(0,6):
        print ("Combo:", combo)
        if combo == 0:
            i,j,k = 0,1,2
        if combo == 1:
            i,j,k = 0,2,1
        if combo == 2:
            i,j,k = 1,0,2
        if combo == 3:
            i,j,k = 1,2,0
        if combo == 4:
            i,j,k = 2,0,1
        if combo == 5:
            i,j,k = 2,1,0
        print ("RGB combo:", i,j,k)

        rgb_map = np.zeros([res,res,3])
        rgb_map[:,:,i] = norm_dens_proj_grid.T
        rgb_map[:,:,j] = norm_pres_proj_grid.T
        rgb_map[:,:,k] = norm_temp_proj_grid.T
        

        if rectangle_mode:
            save_path_ext = save_path+f'rectangle/combo_{combo}/'
            if not os.path.exists(save_path_ext):
                os.makedirs(save_path_ext)
            fig, ax = plt.subplots()
            fig.set_size_inches(16,9)
            p = ax.imshow(rgb_map[int(3.5*res/16):int(12.5*res/16),:,:], origin='lower')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(save_path_ext+f'rgb_galaxy_{snapnum}.jpg', dpi=200)
            #plt.show()
            plt.close()

        save_path_ext = save_path+f'combo_{combo}/'
        if not os.path.exists(save_path_ext):
            os.makedirs(save_path_ext)
        fig, ax = plt.subplots()
        fig.set_size_inches(12,12)
        p = ax.imshow(rgb_map, origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(save_path_ext+f'rgb_galaxy_{snapnum}.jpg', dpi=200)
        #plt.show()
        plt.close()


def load_gal_quants_data(params, snapnum):
    #snapnum = 500
    snapdir = params.path
    print ("Loading data...")
    print (snapdir, snapnum)
    masses = load_fire_data("Masses",0,snapdir,snapnum)
    #print (snapdir, masses.shape)
    coords = load_fire_data("Coordinates",0,snapdir,snapnum)
    hsml = load_fire_data("SmoothingLength",0,snapdir,snapnum)
    pressure = load_fire_data("Pressure",0,snapdir,snapnum)
    dens = load_fire_data("Density",0,snapdir,snapnum)
    #temps = load_fire_data("Temperature",0,snapdir,snapnum)
    #sound_speed = load_fire_data("SoundSpeed",0,snapdir,snapnum)
    #molecular_mass_fraction = load_fire_data("MolecularMassFraction",0,snapdir,snapnum)
    temps = load_fire_snap("Temperature", 0, snapdir, snapnum)
    print ("Loaded data...")
    
    gal_quants0 = GalQuants(params, snapnum, r_gal, h)
    gal_quants0.project(coords)
    #print (gal_quants0.gal_centre)
    #print (gal_quants0.gal_centre_proj)
    #print (gal_quants0.proj)
    gal_quants0.add_key("Masses", masses, 1)
    #gal_quants0.add_key("Velocities", velocities0, 3)
    gal_quants0.add_key("SmoothingLength", hsml, 1)
    gal_quants0.add_key("Density", dens, 1)
    gal_quants0.add_key("Pressure", pressure, 1)
    #gal_quants0.add_key("SoundSpeed", sound_speed, 1)
    #gal_quants0.add_key("MolecularMassFraction", molecular_mass_fraction, 1)
    gal_quants0.add_key("Temperature", temps, 1)

    del masses, coords, hsml, pressure, dens, temps #sound_speed, molecular_mass_fraction, temps
    
    print ("Coords:", gal_quants0.data["Coordinates"].shape)
    print ("Masses:", gal_quants0.data["Masses"].shape)
    print ("SmoothingLength:", gal_quants0.data["SmoothingLength"].shape)
    print ("Density:", gal_quants0.data["Density"].shape)
    print ("Pressure:", gal_quants0.data["Pressure"].shape)
    #print ("SoundSpeed:", gal_quants0.data["SoundSpeed"].shape)
    #print ("MolecularMassFraction:", gal_quants0.data["MolecularMassFraction"].shape)
    print ("Temperature:", gal_quants0.data["Temperature"].shape)


    return gal_quants0













if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    snapnum = convert_to_array(args['--snapnum'])
    r_gal = float(args['--r_gal'])
    h = float(args['--h'])
    #cb_range = convert_to_array(args['--colorbar_range'], dtype=np.float64)
    save_path = path+args['--save_path']
    #gas_data_sub_dir = args['--gas_data_sub_dir']
    #star_data_sub_dir = args['--star_data_sub_dir']
    #image_filename_prefix = args['--img_filename_prefix']
    #image_filename_suffix = args['--img_filename_suffix']
    #img_filename_mode = int(args['--img_filename_mode'])
    image_box_size = float(args['--box_size'])
    rectangle_mode = convert_to_bool(args['--rectangle_mode'])
    all_snaps_in_dir = convert_to_bool(args['--all_snaps_in_dir'])


    image_path = path+'img_data/'

    image_filename_prefix = 'center_proj_'
    image_filename_suffix = '.hdf5'
    
    ## Some bookkeeping
    start_snap = 500    #Dummy if considering only one snapshot
    last_snap = 1000     #Dummy if considering only one snapshot
    filename_prefix = "Linked_Clouds_"
    cloud_num_digits = 4
    snapshot_num_digits = 4
    cloud_prefix = "Cloud"
    
    image_filename_prefix = 'center_proj_'
    image_filename_suffix = '.hdf5'
    hdf5_file_prefix = 'Clouds_'
    age_cut = 1
    dat_file_header_size=8
    snapshot_prefix="Snap"
    cph_sub_dir="CloudPhinderData/"
    gas_data_sub_dir="GasData/"
    star_data_sub_dir="StarData/"
    frac_thresh='thresh0.0'
    nmin = 10
    vir = 5
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
    sim=sim
    image_path = 'img_data/'

    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)



    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "white",
        "axes.facecolor": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})


    if all_snaps_in_dir:
        # Get the list of snapshots in the directory, try a bunch of ways
        snap_list = np.sort(glob.glob(path+'snapshot*.hdf5'))
        if snap_list.size>0:
            snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            snap_list = np.sort(glob.glob(path+'snapdir*/snapshot*.0.hdf5'))
            if snap_list.size>0:
                snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.0.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            snap_list = np.sort(glob.glob(path+'snapdir_*/*.hdf5'))
            if snap_list.size>0:
                snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            print ('No snapshots found in the given directory.')
            # Exit if no snapshots found
            exit()
        
        print ("List of snapnums:", snap_num_list)
        for snap in snap_num_list:
            print ("==============================================================")
            print ("Snapshot: ", snap)
            gal_quants0 = load_gal_quants_data(params, snap)
            plot_rgb_galaxy(gal_quants0, snap, image_box_size, save_path, rectangle_mode)

            print ("==============================================================")

    else:
        if len(snapnum)==1:
            snap = snapnum[0]
            gal_quants0 = load_gal_quants_data(params, snap)
            plot_rgb_galaxy(gal_quants0, snap, image_box_size, save_path, rectangle_mode)
        
        else:
            print ("Snapnum is a range...")
            for snap in range(snapnum[0], snapnum[1]+1):
                print ("==============================================================")
                print ("Snapshot: ", snap)
                gal_quants0 = load_gal_quants_data(params, snap)
                plot_rgb_galaxy(gal_quants0, snap, image_box_size, save_path, rectangle_mode)
                print ("==============================================================")

