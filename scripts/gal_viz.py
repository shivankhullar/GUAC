#!/usr/bin/env python
"""
gal_viz: "Visualize a galaxy snapshot -- plot the surface density of gas particles in a galaxy as seen from a face-on view"

Usage: gal_viz.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: ]
    --snapnum=<snapnum>                                 Snapshot number [default: 600]
    --snapnum_range=<snapnum_range>                     Range of snapshots to process (e.g., 0,100) [default: ]
    --r_gal=<r_gal>                                     Galaxy radius [default: 25]
    --h=<h>                                             Scale height [default: 0.4] 
    --save_path=<save_path>                             Path to save the images [default: img_data/]
    --colorbar_range=<colorbar_range>                   Range for colorbar [default: 1e-2,3e3]
    --gas_data_sub_dir=<gas_data_sub_dir>               Subdirectory containing gas data [default: GasData/]
    --star_data_sub_dir=<star_data_sub_dir>             Subdirectory containing star data [default: StarData/]
    --img_filename_prefix=<img_filename_prefix>         Prefix for image filename [default: center_proj_]
    --img_filename_suffix=<img_filename_suffix>         Suffix for image filename [default: .hdf5]
    --img_filename_mode=<img_filename_mode>             Mode for image filename [default: 0]
    --image_box_size=<image_box_size>                   Size of the image box [default: 35]
    --center_and_proj=<center_and_proj>                 Find center and project galaxy [default: False]
    --num_processes=<num_processes>                     Number of parallel processes [default: 8]
"""

from docopt import docopt
from galaxy_utils.gal_utils import *
from generic_utils.fire_utils import *
from generic_utils.script_utils import *
from meshoid import Meshoid
from matplotlib import colors
from visualization.image_maker import edgeon_faceon_projection
import os
from multiprocessing import Pool
from functools import partial

def make_photo(path, snapnum, save_path):
    edgeon_faceon_projection(path,   #snapshot director
                         snapnum,        #snapshot number
                         field_of_view=35,  #set the size of the image
                         #image_name_prefix='Faceon_{}'.format(snapnum),
                         faceon=True,
                         pixels=2048,
                         output_directory=save_path,
                         just_center_and_project=True)       #do a faceon image (or set faceon=False or edgeon=True to get an edge-on image)
                         #**kwargs)  #kwargs are passed to image_maker, get_center, and load_fire_snap



def plot_galaxy(params, snapnum, r_gal, h, save_path, cb_range, image_box_size=35, res=1024, snapdir=True):
    try:
        key="gas"
        field="Masses"
        print ("Loading data from GasData array files...")
        masses = load_fire_data_arr(key, field, snapnum, params)
        field="Coordinates"
        coords = load_fire_data_arr(key, field, snapnum, params)
        field="SmoothingLength"
        hsml = load_fire_data_arr(key, field, snapnum, params)
    except:
        print ("GasData array files not found... Loading data from snapshot file...")
        if snapdir==True:
            snapdir = params.path+"snapdir_{snapnum}/".format(snapnum=snapnum)
        else:
            print ("Loading data from snapshot file (no snapdir) ...")
            snapdir = params.path
        #print (snapdir, snapnum)
        masses = load_fire_data("Masses",0,snapdir,snapnum)
        #print (snapdir, masses.shape)
        coords = load_fire_data("Coordinates",0,snapdir,snapnum)
        hsml = load_fire_data("SmoothingLength",0,snapdir,snapnum)
        

    gal_quants0 = GalQuants(params, snapnum, r_gal, h)
    gal_quants0.project(coords)
    #print (gal_quants0.gal_centre)
    #print (gal_quants0.gal_centre_proj)
    #print (gal_quants0.proj)
    gal_quants0.add_key("Masses", masses, 1)
    #gal_quants0.add_key("Velocities", velocities0, 3)
    gal_quants0.add_key("SmoothingLength", hsml, 1)


    pos, mass, hsml = gal_quants0.data["Coordinates"], gal_quants0.data["Masses"], \
                        gal_quants0.data["SmoothingLength"]#, gal_quants0.data["Velocities"]

    res = res
    #box_size = 50 #pdata['BoxSize']
    #image_box_size = 35#pdata['BoxSize']
    center = gal_quants0.gal_centre_proj #np.array([box_size/2, box_size/2, box_size/2])

    M = Meshoid(pos, mass, hsml)

    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    #box_size = max_pos-min_pos
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y)
    #fig, ax = plt.subplots(figsize=(10,10))
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)*1e4

    print ('Colorbar range = ', cb_range[0], cb_range[1])
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    sigma_gas_msun_pc2[sigma_gas_msun_pc2<cb_range[0]]=cb_range[0]
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=cb_range[0],vmax=cb_range[1]), cmap='inferno')
    #ax[0].set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    #pixels = res
    #fov = 35

    fontprops = fm.FontProperties(size=18)
    #scalebar = AnchoredSizeBar(ax.transData,
    #                        5*pixels/fov, '5 kpc', 'upper left', 
    #                        pad=1,
    #                        color='white',
    #                        frameon=False,
    #                        size_vertical=5,
    #                        fontproperties=fontprops)
    scalebar = AnchoredSizeBar(ax.transData,
                            5, '5 kpc', 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=0.5,
                            fontproperties=fontprops)

    ax.add_artist(scalebar)

    #cb_ax = fig.add_axes([0.9, 0.125, 0.01, 0.755])
    #cbar = fig.colorbar(p, cax=cb_ax, orientation="vertical",\
    #                    ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])#, label='Counts', labelsize=22)

    #cbar.set_label(label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$", size=22)
    #cbar = fig.colorbar(h[3], ax=ax[0,1])#, labelsize=20)
    #cbar.ax.tick_params(labelsize=20)#, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=0, labelbottom=False, bottom=False, top=False, labeltop=False, left=False, labelleft=False) 
    ax.tick_params(axis='both', which='minor', labelsize=0, labelbottom=False, bottom=False, top=False, labeltop=False, left=False, labelleft=False) 
    plt.tight_layout()
    # Make save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path+'proj_{snap:04d}.png'.format(snap=snapnum), dpi=300)
    print ("Saved image to {}".format(save_path+'proj_{snap:04d}.png'.format(snap=snapnum)))
    #plt.show()
    plt.close()

    return


def process_snapshot_viz(snapnum, path, sim, r_gal, h, save_path, cb_range, image_box_size, snapdir, 
                         gas_data_sub_dir, star_data_sub_dir, image_filename_prefix, 
                         image_filename_suffix, img_filename_mode):
    """Wrapper function for parallel processing"""
    try:
        print("==============================================================")
        print("Processing snapshot: ", snapnum)
        
        # Setup params for this snapshot
        start_snap = 591
        last_snap = 614
        filename_prefix = "Linked_Clouds_"
        cloud_num_digits = 4
        snapshot_num_digits = 4
        cloud_prefix = "Cloud"
        hdf5_file_prefix = 'Clouds_'
        age_cut = 1
        dat_file_header_size = 8
        snapshot_prefix = "Snap"
        cph_sub_dir = "CloudPhinderData/"
        frac_thresh = 'thresh0.0'
        nmin = 10
        vir = 5
        sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
        image_path = 'img_data/'
        
        params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits,
                       snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut,
                       dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,
                       image_path, image_filename_prefix,
                       image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)
        
        plot_galaxy(params, snapnum, r_gal, h, save_path, cb_range, image_box_size=image_box_size, snapdir=snapdir)
        
        print("Completed snapshot: ", snapnum)
        print("==============================================================")
        return snapnum, True
    except Exception as e:
        print(f"Error processing snapshot {snapnum}: {e}")
        return snapnum, False


if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    snapnum = int(args['--snapnum'])
    snapnum_range = args['--snapnum_range']
    r_gal = float(args['--r_gal'])
    h = float(args['--h'])
    cb_range = convert_to_array(args['--colorbar_range'], dtype=np.float64)
    save_path = path+args['--save_path']
    gas_data_sub_dir = args['--gas_data_sub_dir']
    star_data_sub_dir = args['--star_data_sub_dir']
    image_filename_prefix = args['--img_filename_prefix']
    image_filename_suffix = args['--img_filename_suffix']
    img_filename_mode = int(args['--img_filename_mode'])
    image_box_size = int(args['--image_box_size'])
    num_processes = int(args['--num_processes'])

    image_path = path+'img_data/'

    if args['--center_and_proj'] == 'True':
        print("Finding center and projection matrix...")
        # If snapnum_range is provided, process range
        if snapnum_range:
            snap_range = convert_to_array(snapnum_range, dtype=np.int32)
            if len(snap_range) == 2:
                for snap in range(snap_range[0], snap_range[1] + 1):
                    make_photo(path, snap, save_path)
            else:
                print("Invalid snapnum_range format. Use: start,end")
                exit(1)
        else:
            make_photo(path, snapnum, save_path)
        img_filename_mode = 1
        image_path = save_path
    else:
        print("Using center and projection projection from existing file...")
        pass

    # 0: Do nothing, use default
    # 1: Use center_proj_*.hdf5
    # 2: Use image_faceon_s0*_fov0035_Ngb32_star.hdf5
    if img_filename_mode == 2:
        image_filename_prefix = 'image_faceon_s0'
        image_filename_suffix = '_fov0035_Ngb32_star.hdf5'
    elif img_filename_mode == 1:
        image_filename_prefix = 'center_proj_'
        image_filename_suffix = '.hdf5'
    elif img_filename_mode == 0:
        pass
    else:
        print("Invalid image filename mode, should be 0, 1 or 2...")
        exit(1)

    # Determine which snapshots to process
    snap_num_list = None
    if snapnum_range:
        snap_range = convert_to_array(snapnum_range, dtype=np.int32)
        if len(snap_range) == 2:
            snap_num_list = np.arange(snap_range[0], snap_range[1] + 1)
            print(f"Processing snapshot range: {snap_range[0]} to {snap_range[1]}")
        else:
            print("Invalid snapnum_range format. Use: start,end")
            exit(1)
    else:
        snap_num_list = [snapnum]

    # Process snapshots
    if len(snap_num_list) > 1:
        print(f"Processing {len(snap_num_list)} snapshots using {num_processes} parallel processes...")
        
        # Create partial function with fixed parameters
        process_func = partial(process_snapshot_viz, 
                              path=path, sim=sim, r_gal=r_gal, h=h, 
                              save_path=save_path, cb_range=cb_range, 
                              image_box_size=image_box_size, snapdir=snapdir,
                              gas_data_sub_dir=gas_data_sub_dir, 
                              star_data_sub_dir=star_data_sub_dir,
                              image_filename_prefix=image_filename_prefix, 
                              image_filename_suffix=image_filename_suffix,
                              img_filename_mode=img_filename_mode)
        
        # Use multiprocessing pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, snap_num_list)
        
        # Report results
        successful = sum(1 for _, success in results if success)
        print(f"\nCompleted: {successful}/{len(snap_num_list)} snapshots processed successfully")
    else:
        # Single snapshot processing
        ## Some bookkeeping
        start_snap = 591
        last_snap = 614
        filename_prefix = "Linked_Clouds_"
        cloud_num_digits = 4
        snapshot_num_digits = 4
        cloud_prefix = "Cloud"
        hdf5_file_prefix = 'Clouds_'
        age_cut = 1
        dat_file_header_size = 8
        snapshot_prefix = "Snap"
        cph_sub_dir = "CloudPhinderData/"
        frac_thresh = 'thresh0.0'
        nmin = 10
        vir = 5
        sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
        image_path = 'img_data/'

        params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits,
                       snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut,
                       dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,
                       image_path, image_filename_prefix,
                       image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)

        plot_galaxy(params, snapnum, r_gal, h, save_path, cb_range, image_box_size=image_box_size, snapdir=snapdir)
