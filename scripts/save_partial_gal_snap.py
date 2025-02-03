#!/usr/bin/env python
"""
save_partial_gal_snap.py: "Create initial conditions file for FIRE+STARFORGE simulation 
                            with a refinement particle placed at a custom position.
                            The custom position is specified by the user. The script will
                            then choose particles within a certain distance of the custom
                            position to calculate the center of mass velocity of the SMBH particle.
                            One can specify if the particles should follow stars or gas particles while
                            deciding the original position for the refinement particle.
                            The script will then create an hdf5 file with the ICs."

Usage: save_partial_gal_snap.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: ./]
    --save_path=<save_path>                             Path to save the images [default: ./]
    --image_box_size=<image_box_size>                   Size of the image box [default: 15]
    --snapnum_range=<snapnum_range>                     Range of snapshots to plot [default: 0,100]
    --res=<res>                                         Resolution of the image [default: 2048]
"""


from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from generic_utils.load_fire_snap import *
from galaxy_utils.gal_utils import *
import time



import glob
#import yt
import h5py
import numpy as np
from docopt import docopt




def save_data(save_path, name, data, snapnum):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = save_path + name + "_snap" + str(snapnum) + ".npy"
    np.save(filename, data)


def process_snapshot(params, snapnum, save_path):
    start = time.time()

    print("Processing snapshot:", snapnum)
    snapdir = params.path
    print ("snapdir: ", snapdir)
    masses = load_fire_data("Masses", 0, snapdir, snapnum)
    coords = load_fire_data("Coordinates", 0, snapdir, snapnum)
    hsml = load_fire_data("SmoothingLength", 0, snapdir, snapnum)
    vels = load_fire_data("Velocities", 0, snapdir, snapnum)
    temps = load_fire_snap("Temperature", 0, snapdir, snapnum)
    pres = load_fire_data("Pressure", 0, snapdir, snapnum)
    dens = load_fire_data("Density", 0, snapdir, snapnum)

    print("Data loaded")
    
    gal_quants0 = GalQuants(params, snapnum, r_gal, h)
    gal_quants0.project(coords)
    gal_quants0.add_key("Masses", masses, 1)
    gal_quants0.add_key("Velocities", vels, 3)
    gal_quants0.add_key("SmoothingLength", hsml, 1)
    gal_quants0.add_key("Temperature", temps, 1)
    gal_quants0.add_key("Pressure", pres, 1)
    gal_quants0.add_key("Density", dens, 1)

    
    print ("Galaxy quantities created", len(gal_quants0.data["Masses"]))
    save_data(save_path, "gal_quants0", gal_quants0, snapnum)
    print("Time taken for this snapshot:", time.time() - start)
    
    del masses, coords, hsml, vels

    return




if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    save_path = path + sim + '/' + args['--save_path']
    print ("save_path: ", save_path)
    image_box_size = float(args['--image_box_size'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    cut_off_distance = 1.0

    start_snap = snapnum_range[0]
    last_snap = snapnum_range[1]
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
    gas_data_sub_dir = "GasData/"
    cph_sub_dir="CloudPhinderData/"
    #cph_sub_dir='m12i_restart/'
    frac_thresh='thresh0.3'

    r_gal = 25
    h = 3 #0.4

    #save_path = './data/'
    #snapnum = 650

    nmin = 10
    vir = 5
    sim = args['--sim']
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
    filename = path+sub_dir+filename_prefix+str(start_snap)+"_"+str(last_snap)+"names"+".txt"


    
    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)
        


    for snapnum in range(snapnum_range[0], snapnum_range[1] + 1):
        process_snapshot(params, snapnum, save_path=save_path)

