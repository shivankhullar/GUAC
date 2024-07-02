#!/usr/bin/env python
"""
ism_viz: "Visualize patch of a galaxy"

Usage: ism_viz.py [options]

Options:
    -h, --help                                                  Show this screen
    --path=<path>                                               Path to the simulation directory [default: ./]
    --sim=<sim>                                                 Simulation name [default: m12i_final_fb_7k]
    --snapdir=<snapdir>                                         Are snapshots in a snapdir directory? [default: True]
    --snapnum=<snapnum>                                         Snapshot number [default: 600]
    --all_snaps_in_dir=<all_snaps_in_dir>                       Are all snapshots in the same directory? [default: False]
    --r_gal=<r_gal>                                             Galaxy radius [default: 25]
    --h=<h>                                                     Scale height [default: 0.4] 
    --save_path=<save_path>                                     Path to save the images [default: ./]
    --dist=<dist>                                               Distance from the center [default: 3]
    --special_position_flag=<special_position_flag>             Box around a special position? See below [default: False]
    --special_position_custom=<special_custom_position>         Box around a special position? Enter coords manually. [default: 0,0,0]
    --special_position_refine=<special_position_refine>         Box around the refinement particle? [default: False]
    --special_position_refine_fix=<special_position_refine_fix> Box around the refinement particle at first snap? [default: False]
    --box_size=<box_size>                                       Size of the image box [default: 7]
    --age_cut=<age_cut>                                         Age cut [default: 3]
    --stars=<stars>                                             Include stars? [default: False]
    --res=<res>                                                 Resolution of the image [default: 1024]
"""


from docopt import docopt
from galaxy_utils.gal_utils import *
from generic_utils.fire_utils import *
from generic_utils.script_utils import *
import glob
import yt
import h5py
from meshoid import Meshoid
#matplotlib.use('Agg')
from matplotlib import colors
import colorcet as cc
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


#from visualization import *
#from visualization.image_maker import edgeon_faceon_projection
#import os


def get_stellar_ages(sft, params, snapnum, snapdir=None):
    if snapdir==None:
        snapdir = params.path+"snapdir_{num}/".format(num=snapnum)
    try:
        f = h5py.File(snapdir+"snapshot_{num}.hdf5".format(num=snapnum), 'r')
    except:
        try:
            f = h5py.File(snapdir+"snapshot_{num}.0.hdf5".format(num=snapnum), 'r')
        except:
            print("Snapshot file not found.")
            return
    current_a = f['Header'].attrs['Time']
    try:
        omega_matter = f['Header'].attrs['Omega0']
        omega_lambda = f['Header'].attrs['OmegaLambda']
        hubble_constant = f['Header'].attrs['HubbleParam']
        co = yt.utilities.cosmology.Cosmology(omega_matter=omega_matter, omega_lambda=omega_lambda, hubble_constant=hubble_constant)
    except:    
        co = yt.utilities.cosmology.Cosmology(omega_matter=0.272, omega_lambda=0.728, hubble_constant=0.702)
    
    current_time = co.t_from_a(current_a).in_units('Myr').value
    sft_myr = co.t_from_a(sft).in_units('Myr').value
    ages_myr = current_time - sft_myr
    return sft_myr, ages_myr



def make_plot(gal_quants0, distance_from_center, image_box_size, res, save_path, dist, age_cut, gal_quants4, refine_pos=None, special_position=False):
    pos, mass, hsml = gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], \
                            gal_quants0.data["SmoothingLength"]#, gal_quants0.data["Velocities"]

    
    center = gal_quants0.gal_centre_proj #np.array([box_size/2, box_size/2, box_size/2])
    
    #dist = 3
    #center = center + np.array([dist,dist,0])
    center = center + distance_from_center   #np.array([x_dist,y_dist,z_dist])
    print ("Center:", center)
    M = Meshoid(pos, mass, hsml)

    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    #box_size = max_pos-min_pos
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y, indexing='ij')
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)*1e4

    plt.rcParams.update({"lines.color": "white",
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

    fig, ax = plt.subplots(figsize=(10,10))
    cmap = get_cmap("cet_fire")
    #cmap = get_cmap("cet_bmy")

    #cmap = plt.cm.get_cmap('magma')

    image = sigma_gas_msun_pc2
    image[image<1e-3] = 1e-3

    p = ax.pcolormesh(X, Y, image, norm=colors.LogNorm(vmin=5e-1,vmax=5e3), cmap=cmap)
    #ax[0].set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])

    fontprops = fm.FontProperties(size=18)
    
    # Create scalebar according to image_box_size
    # Currently finding the nearest multiple of 100 for box sizes < 4 kpc
    # and nearest multiple of 1 for box sizes > 4 kpc
    fraction_of_box_in_scalebar = 0.25
    scale = 1/fraction_of_box_in_scalebar
    if image_box_size>=scale:
        scale_bar_length = image_box_size//scale
        scale_bar_string = str(scale_bar_length)+'kpc'
    elif image_box_size<scale and image_box_size>scale/10:
        scale_bar_length = image_box_size/scale*1000//100
        scale_bar_string = str(scale_bar_length)+'pc'
    elif image_box_size<scale/10 and image_box_size>scale/100:
        scale_bar_length = image_box_size/scale*1000//10
        scale_bar_string = str(scale_bar_length)+'pc'
    else:
        scale_bar_length = image_box_size/scale*1000
        scale_bar_string = str(scale_bar_length)+'pc'

    scalebar = AnchoredSizeBar(ax.transData,
                            scale_bar_length, scale_bar_string, 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=scale_bar_length/10,
                            fontproperties=fontprops)

    ax.add_artist(scalebar)

    #plt.scatter(cloud_centres[:,0], cloud_centres[:,1], s=cloud_reffs*1000, c='b', alpha=0.5)
    #plt.scatter(cloud_centres[cloud_num, 0], cloud_centres[cloud_num, 1], s=cloud_reffs[cloud_num]*1500, c='r', alpha=0.5)

    #ax.scatter(star_coords[:,0]+gal_quants4.gal_centre_proj[0], star_coords[:,1]+gal_quants4.gal_centre_proj[1], s=10, c='b', alpha=0.5)
    #age_cut = 3 
    if gal_quants4 is not None:
        ax.scatter(gal_quants4.data["Coordinates"][:,0][gal_quants4.data["Ages"]<age_cut], gal_quants4.data["Coordinates"][:,1][gal_quants4.data["Ages"]<age_cut], s=10, \
                c=gal_quants4.data["Ages"][gal_quants4.data["Ages"]<age_cut], cmap = cm.get_cmap('Blues_r'), alpha=1)
        #ax.scatter(gal_quants4.data["Coordinates"][:,0][::300], gal_quants4.data["Coordinates"][:,1][::300], s=0.5, c='w')
    if refine_pos is not None:
        ax.scatter(refine_pos[0], refine_pos[1], s=10, c='g')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if special_position:
        if gal_quants4:
            save_filename = save_path+'ism_patch_{dist}_{size}_{age_cut}_{snapnum}.jpg'.format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut)
        else:
            save_filename = save_path+'ism_patch_customloc_{size}_{snapnum}.jpg'.format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut)
    else:
        if gal_quants4:
            save_filename = save_path+'ism_patch_{dist}_{size}_{age_cut}_{snapnum}.jpg'.format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut)
        else:
            save_filename = save_path+'ism_patch_{dist}_{size}_{snapnum}.jpg'.format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut)
    
    plt.savefig(save_filename, dpi=200)
    plt.close()

    print (f"ISM patch saved at {save_filename}")



def get_galquants_data(params, snapnum, snapdir_bool, stars, special_refine_pos):
    # Start loading data below
    if snapdir_bool:
        snapdir = params.path+"snapdir_{num}/".format(num=snapnum)
    else:
        snapdir = params.path

    positions0 = load_from_snapshot.load_from_snapshot("Coordinates", 0, snapdir, snapnum, units_to_physical=True)
    masses0 = load_from_snapshot.load_from_snapshot("Masses", 0, snapdir, snapnum, units_to_physical=True)
    hsml0 = load_from_snapshot.load_from_snapshot("SmoothingLength", 0, snapdir, snapnum, units_to_physical=True)
    print ("Loaded gas data...")
    
    gal_quants0 = GalQuants(params, snapnum, r_gal, h)
    gal_quants0.project(positions0)
    gal_quants0.add_key("Masses", masses0, 1)
    #gal_quants0.add_key("Velocities", velocities0, 3)
    gal_quants0.add_key("SmoothingLength", hsml0, 1)
    
    gal_quants3, gal_quants4 = None, None
    if stars:
        positions4 = load_from_snapshot.load_from_snapshot("Coordinates", 4, snapdir, snapnum, units_to_physical=True)
        masses4 = load_from_snapshot.load_from_snapshot("Masses", 4, snapdir, snapnum)
        sft4 = load_from_snapshot.load_from_snapshot("StellarFormationTime", 4, snapdir, snapnum, units_to_physical=True)
        print ("Loaded stellar data...")

        sfts, ages = get_stellar_ages(sft4, params, snapnum, snapdir)    
        gal_quants4 = GalQuants(params, snapnum, r_gal, h)
        gal_quants4.project(positions4)
        gal_quants4.add_key("Masses", masses4, 1)
        gal_quants4.add_key("StellarFormationTime", sfts, 1)
        gal_quants4.add_key("Ages", ages, 1)

    if special_refine_pos:
        positions3 = load_from_snapshot.load_from_snapshot("Coordinates", 3, snapdir, snapnum, units_to_physical=True)
        masses3 = load_from_snapshot.load_from_snapshot("Masses", 3, snapdir, snapnum, units_to_physical=True)
        vels3 =  load_from_snapshot.load_from_snapshot("Velocities", 3, snapdir, snapnum, units_to_physical=True)
        print ("Loaded refinement data...")
        #if snapnum<10:
        #    f = h5py.File(snapdir+"snapshot_00{num}.hdf5".format(num=snapnum), 'r')
        #elif snapnum>=10 and snapnum<100:
        #    f = h5py.File(snapdir+"snapshot_0{num}.hdf5".format(num=snapnum), 'r')
        #else:
        #    f = h5py.File(snapdir+"snapshot_{num}.hdf5".format(num=snapnum), 'r')

        #hubble = f["Header"].attrs["HubbleParam"]
        #hinv = 1./hubble; ascale=f["Header"].attrs["Time"]
        #rconv = ascale*hinv
        #positions3 = np.array(f["PartType3/Coordinates"][:])
        #positions3 *= rconv
        #masses3 = np.array(f["PartType3/Masses"][:])
        #masses3 *= hinv
        #vels3 = np.array(f["PartType3/Velocities"][:])
        #vels3 *= np.sqrt(ascale)

        print (positions3, masses3, vels3, positions3.shape, positions0.shape)
        gal_quants3 = GalQuants(params, snapnum, r_gal, h)
        gal_quants3.project(positions3)
        gal_quants3.add_key("Coordinates", positions3, 3)
        gal_quants3.add_key("Masses", masses3, 1)
        gal_quants3.add_key("Velocities", vels3, 3)
    
    return gal_quants0, gal_quants3, gal_quants4


if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    snapdir = convert_to_bool(args['--snapdir'])
    snapnum = int(args['--snapnum'])
    r_gal = float(args['--r_gal'])
    h = float(args['--h'])
    save_path = path+args['--save_path']
    age_cut = float(args['--age_cut'])
    res = int(args['--res'])
    dist = float(args['--dist'])
    box_size = float(args['--box_size'])
    sim = args['--sim']
    stars = convert_to_bool(args['--stars'])
    all_snaps_in_dir = convert_to_bool(args['--all_snaps_in_dir'])
    special_pos_flag = convert_to_bool(args['--special_position_flag'])
    special_custom_pos = convert_to_array(args['--special_position_custom'], np.float64)
    special_refine_pos = convert_to_bool(args['--special_position_refine'])
    special_refine_pos_fix = convert_to_bool(args['--special_position_refine_fix'])

    ## Some bookkeeping
    start_snap = 591    #Dummy if considering only one snapshot
    last_snap = 614     #Dummy if considering only one snapshot
    filename_prefix = "Linked_Clouds_"
    cloud_num_digits = 4
    snapshot_num_digits = 4
    cloud_prefix = "Cloud"
    
    gas_data_sub_dir = "GasData/"
    star_data_sub_dir = "StarData/"
    image_path = "img_data/"
    image_filename_prefix = 'center_proj_'
    image_filename_suffix = '.hdf5'
    hdf5_file_prefix = 'Clouds_'
    #age_cut = 1
    dat_file_header_size=8
    snapshot_prefix="Snap"
    cph_sub_dir="CloudPhinderData/"
    frac_thresh='thresh0.0'
    nmin = 10
    vir = 5
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)


    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)
        

    

    
    

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
    
        
        for snapnum in snap_num_list:
            print ("Loading data from snapshot {num}...".format(num=snapnum))
            gal_quants0, gal_quants3, gal_quants4 = get_galquants_data(params, snapnum, snapdir, stars, special_refine_pos)

            refine_pos_proj = gal_quants3.data["Coordinates"][0]
            print (refine_pos_proj) 
            if special_refine_pos_fix:
                # Use the original refinement particle as the special position
                if snapnum==snap_num_list[0]:
                    special_position = True
                    distance_from_center = refine_pos_proj - gal_quants0.gal_centre_proj
            else:
                # Use the refinement particle as the special position
                distance_from_center = refine_pos_proj - gal_quants0.gal_centre_proj

            print ("Making plot...")
            make_plot(gal_quants0, distance_from_center, box_size, res, save_path, dist, age_cut, gal_quants4, refine_pos_proj)
            print ("Done!")

    else:
        print ("Loading data from snapshot {num}...".format(num=snapnum))
    
        gal_quants0, gal_quants3, gal_quants4 = get_galquants_data(params, snapnum, snapdir, stars, special_refine_pos)

        # Focus on a special position in the galaxy
        
        proj_special_pos = np.matmul(gal_quants0.proj_matrix, special_custom_pos)
        distance_from_center = proj_special_pos - gal_quants0.gal_centre_proj

        print ("Making plot...")
        make_plot(gal_quants0, distance_from_center, box_size, res, save_path, dist, age_cut, gal_quants4, special_position=False)
        print ("Done!")
