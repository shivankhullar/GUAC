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
    --particle_type=<particle_type>                     Particle type to plot [default: 0]
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


def process_snapshot(params, snapnum, save_path, particle_type=0):
    start = time.time()

    print("Processing snapshot:", snapnum)
    snapdir = params.path
    print ("snapdir: ", snapdir)
    if particle_type==0:
        masses = load_fire_data("Masses", particle_type, snapdir, snapnum)
        coords = load_fire_data("Coordinates", particle_type, snapdir, snapnum)
        hsml = load_fire_data("SmoothingLength", particle_type, snapdir, snapnum)
        vels = load_fire_data("Velocities", particle_type, snapdir, snapnum)
        temps = load_fire_snap("Temperature", particle_type, snapdir, snapnum)
        int_energy = load_fire_snap("InternalEnergy", particle_type, snapdir, snapnum)
        
        pres = load_fire_data("Pressure", particle_type, snapdir, snapnum)
        dens = load_fire_data("Density", particle_type, snapdir, snapnum)
        pIDs = load_fire_data("ParticleIDs", particle_type, snapdir, snapnum)
        pIDgen = load_fire_data("ParticleIDGenerationNumber", particle_type, snapdir, snapnum)
        pIDchilds = load_fire_data("ParticleChildIDsNumber", particle_type, snapdir, snapnum)

        print("Data loaded")
        
        gal_quants0 = GalQuants(params, snapnum, params.r_gal, params.h)
        gal_quants0.project(coords)
        gal_quants0.add_key("Masses", masses, 1)
        gal_quants0.add_key("Velocities", vels, 3)
        gal_quants0.add_key("SmoothingLength", hsml, 1)
        gal_quants0.add_key("Temperature", temps, 1)
        gal_quants0.add_key("InternalEnergy", int_energy, 1)
        gal_quants0.add_key("Pressure", pres, 1)
        gal_quants0.add_key("Density", dens, 1)
        gal_quants0.add_key("ParticleIDs", pIDs, 1)
        gal_quants0.add_key("ParticleIDGenerationNumber", pIDgen, 1)
        gal_quants0.add_key("ParticleChildIDsNumber", pIDchilds, 1)

        try:
            sound_speed = load_fire_snap("SoundSpeed", particle_type, snapdir, snapnum)
            gal_quants0.add_key("SoundSpeed", sound_speed, 1)
        except:
            pass
    
        print ("Galaxy quantities created", len(gal_quants0.data["Masses"]))
        save_data(save_path, f"gal_quants{particle_type}", gal_quants0, snapnum)
        del masses, coords, hsml, vels, dens, temps, pres, pIDs, pIDgen, pIDchilds, int_energy

    if particle_type==4:
        masses = load_fire_data("Masses", particle_type, snapdir, snapnum)
        coords = load_fire_data("Coordinates", particle_type, snapdir, snapnum)
        vels = load_fire_data("Velocities", particle_type, snapdir, snapnum)
        pIDs = load_fire_data("ParticleIDs", particle_type, snapdir, snapnum)
        pIDgen = load_fire_data("ParticleIDGenerationNumber", particle_type, snapdir, snapnum)
        pIDchilds = load_fire_data("ParticleChildIDsNumber", particle_type, snapdir, snapnum)
        sft = load_fire_data("StellarFormationTime", particle_type, snapdir, snapnum)

        print("Data loaded")
        
        gal_quants4 = GalQuants(params, snapnum, params.r_gal, params.h)
        gal_quants4.project(coords)
        gal_quants4.add_key("Masses", masses, 1)
        gal_quants4.add_key("Velocities", vels, 3)
        gal_quants4.add_key("StellarFormationTime", sft, 1)
        gal_quants4.add_key("ParticleIDs", pIDs, 1)
        gal_quants4.add_key("ParticleIDGenerationNumber", pIDgen, 1)
        gal_quants4.add_key("ParticleChildIDsNumber", pIDchilds, 1)

        print ("Galaxy quantities created", len(gal_quants4.data["Masses"]))
        save_data(save_path, f"gal_quants{particle_type}", gal_quants4, snapnum)
        del masses, coords, vels, pIDs, pIDgen, pIDchilds, sft

    print("Time taken for this snapshot:", time.time() - start)
    
    return




if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    save_path = args['--save_path'] #path + sim + '/' + args['--save_path']
    print ("save_path: ", save_path)
    image_box_size = float(args['--image_box_size'])
    snapnum_range = convert_to_array(args['--snapnum_range'], dtype=np.int32)
    part_type = int(args['--particle_type'])
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


    
    #params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
    #                snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
    #                dat_file_header_size, gas_data_sub_dir, star_data_sub_dir, cph_sub_dir,\
    #                image_path, image_filename_prefix,\
    #                image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)
    vels_sub_dir = 'vel_sub/'
    gal_quants_sub_dir = 'gal_quants/'
    phinder_sub_dir="VoidPhinderData/"
    linked_filename_prefix = "Linked_Voids_"



    params = Params(path=path, sub_dir=sub_dir, start_snap=start_snap, last_snap=last_snap, filename_prefix=linked_filename_prefix, \
                    cloud_prefix=cloud_prefix, hdf5_file_prefix=hdf5_file_prefix, frac_thresh=frac_thresh, sim=sim, r_gal=r_gal, h=h,\
                    gal_quants_sub_dir=gal_quants_sub_dir, vels_sub_dir=vels_sub_dir, phinder_sub_dir=phinder_sub_dir)
    


    for snapnum in range(snapnum_range[0], snapnum_range[1] + 1):
        process_snapshot(params, snapnum, save_path=save_path, particle_type=part_type)

