#!/usr/bin/env python
"""
ism_viz: "Visualize patch of a galaxy"

Usage: ism_viz.py [options]

Options:
    -h, --help                  Show this screen
    --path=<path>               Path to the simulation directory [default: ./]
    --sim=<sim>                 Simulation name [default: m12i_final_fb_7k]
    --snapdir=<snapdir>         Are snapshots in a snapdir directory? [default: True]
    --snapnum=<snapnum>         Snapshot number [default: 600]
    --r_gal=<r_gal>             Galaxy radius [default: 25]
    --h=<h>                     Scale height [default: 0.4] 
    --save_path=<save_path>     Path to save the images [default: ./]
    --dist=<dist>               Distance from the center [default: 3]
    --box_size=<box_size>       Size of the image box [default: 7]
    --age_cut=<age_cut>         Age cut [default: 3]
    --res=<res>                 Resolution of the image [default: 2048]
"""


from docopt import docopt
import sys
sys.path.insert(0, '../src/')
from fire_utils import *
from gal_quants_funcs import *
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



if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    snapdir = args['--snapdir']
    snapnum = int(args['--snapnum'])
    r_gal = float(args['--r_gal'])
    h = float(args['--h'])
    save_path = path+args['--save_path']
    age_cut = float(args['--age_cut'])
    res = int(args['--res'])
    dist = float(args['--dist'])
    box_size = float(args['--box_size'])
    sim = args['--sim']

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
                    dat_file_header_size, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim)
        

    # Start loading data below
    if snapdir == "True":
        snapdir = params.path+"snapdir_{num}/".format(num=snapnum)
    else:
        snapdir = params.path

    print ("Loading data from snapshot {num}...".format(num=snapnum))
    
    positions0 = load_from_snapshot.load_from_snapshot("Coordinates", 0, snapdir, snapnum)
    masses0 = load_from_snapshot.load_from_snapshot("Masses", 0, snapdir, snapnum)
    hsml0 = load_from_snapshot.load_from_snapshot("SmoothingLength", 0, snapdir, snapnum)
    print ("Loaded gas data...")

    positions4 = load_from_snapshot.load_from_snapshot("Coordinates", 4, snapdir, snapnum)
    masses4 = load_from_snapshot.load_from_snapshot("Masses", 4, snapdir, snapnum)
    sft4 = load_from_snapshot.load_from_snapshot("StellarFormationTime", 4, snapdir, snapnum)
    print ("Loaded stellar data...")

    sfts, ages = get_stellar_ages(sft4, params, snapnum, snapdir)

    gal_quants0 = GalQuants(params, snapnum, r_gal, h)
    gal_quants0.project(positions0)
    gal_quants0.add_key("Masses", masses0, 1)
    #gal_quants0.add_key("Velocities", velocities0, 3)
    gal_quants0.add_key("SmoothingLength", hsml0, 1)

    gal_quants4 = GalQuants(params, snapnum, r_gal, h)
    gal_quants4.project(positions4)
    gal_quants4.add_key("Masses", masses4, 1)
    gal_quants4.add_key("StellarFormationTime", sfts, 1)
    gal_quants4.add_key("Ages", ages, 1)





    pos = gal_quants0.data["Coordinates"]
    #center = np.median(pos,axis=0)
    #pos -= center
    #radius_cut = np.sum(pos*pos,axis=1) < 40*40
    pos, mass, hsml = pos, gal_quants0.data["Masses"], \
                            gal_quants0.data["SmoothingLength"]#, gal_quants0.data["Velocities"]

    image_box_size = box_size #pdata['BoxSize']
    center = gal_quants0.gal_centre_proj #np.array([box_size/2, box_size/2, box_size/2])
    
    #dist = 3
    center = center + np.array([dist,dist,0])
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


    pixels = res
    fov = 35

    fontprops = fm.FontProperties(size=18)
    #scalebar = AnchoredSizeBar(ax.transData,
    #                        5*pixels/fov, '5 kpc', 'upper left', 
    #                        pad=1,
    #                        color='white',
    #                        frameon=False,
    #                        size_vertical=5,
    #                        fontproperties=fontprops)
    scalebar = AnchoredSizeBar(ax.transData,
                            1, '1 kpc', 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=0.1,
                            fontproperties=fontprops)

    ax.add_artist(scalebar)

    #plt.scatter(cloud_centres[:,0], cloud_centres[:,1], s=cloud_reffs*1000, c='b', alpha=0.5)
    #plt.scatter(cloud_centres[cloud_num, 0], cloud_centres[cloud_num, 1], s=cloud_reffs[cloud_num]*1500, c='r', alpha=0.5)

    #ax.scatter(star_coords[:,0]+gal_quants4.gal_centre_proj[0], star_coords[:,1]+gal_quants4.gal_centre_proj[1], s=10, c='b', alpha=0.5)
    #age_cut = 3
    ax.scatter(gal_quants4.data["Coordinates"][:,0][gal_quants4.data["Ages"]<age_cut], gal_quants4.data["Coordinates"][:,1][gal_quants4.data["Ages"]<age_cut], s=10, \
            c=gal_quants4.data["Ages"][gal_quants4.data["Ages"]<age_cut], cmap = cm.get_cmap('Blues_r'), alpha=1)
    ax.scatter(gal_quants4.data["Coordinates"][:,0][::300], gal_quants4.data["Coordinates"][:,1][::300], s=0.5, c='w')
    #ax.scatter(-32040, 10600, s=10, c='g', alpha=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path+'ism_patch_{dist}_{size}_{age_cut}_{snapnum}.jpg'.format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut), dpi=200)
    plt.close()

    print ("ISM patch saved at {path}".format(path=save_path+"ism_patch_{dist}_{size}_{age_cut}_{snapnum}.jpg".format(snapnum=snapnum, dist=int(dist), size=int(image_box_size), age_cut=age_cut)))