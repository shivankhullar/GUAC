#!/usr/bin/env python
"""
create_fire_ic_custom_pos: "Create initial conditions file for FIRE+STARFORGE simulation 
                            with a refinement particle placed at a custom position.
                            The custom position is specified by the user. The script will
                            then choose particles within a certain distance of the custom
                            position to calculate the center of mass velocity of the SMBH particle.
                            One can specify if the particles should follow stars or gas particles while
                            deciding the original position for the refinement particle.
                            The script will then create an hdf5 file with the ICs."

Usage: create_fire_ic_custom_pos.py [options]

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --threshold=<threshold>                             Threshold for cloud selection [default: 0.3]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: m12i_final_fb_7k]
    --ic_path=<ic_path>                                 Path to save the IC file [default: ./] 
    --dist_cut_off=<dist_cut_off>                       Distance cut off for gas particles to be used in finding COM velocity (in kpc) [default: 0.1]
    --follow_particle_types=<follow_particle_types>     Particle types to follow in the simulation [default: 0,4]
    --custom_refine_pos=<custom_refine_pos>             Custom position for the refinement particle [default: 0,0,0]
    --refine_pos_snap=<refine_pos_snap>                 Snapshot number corresponding to the custom position of the refinement particle [default: 600]
    --starting_snap=<starting_snap>                     Starting snapshot number [default: 600]
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from docopt import docopt
import numpy as np
import os
import yt
from yt.utilities.cosmology import Cosmology


def load_snapshot_data(params, snap_num, ptype, units_to_physical=True):
    #Need cosmo units so set units_to_physical=False
    #units_to_physical = False
    print ("Loading data for snapshot {num}".format(num=snap_num))
    snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
    coords = load_from_snapshot.load_from_snapshot('Coordinates', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    masses = load_from_snapshot.load_from_snapshot('Masses', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    vels = load_from_snapshot.load_from_snapshot('Velocities', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    pIDs = load_from_snapshot.load_from_snapshot('ParticleIDs', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    pID_gen_nums = load_from_snapshot.load_from_snapshot('ParticleIDGenerationNumber', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    pID_child_nums = load_from_snapshot.load_from_snapshot('ParticleChildIDsNumber', ptype, snapdir, snap_num, units_to_physical=units_to_physical)
    pID_array = np.column_stack((pIDs, pID_gen_nums, pID_child_nums))

    if ptype == 0:
        #dens = load_from_snapshot.load_from_snapshot('Density', 0, snapdir, snap_num, units_to_physical=units_to_physical)
        #hsmls = load_from_snapshot.load_from_snapshot('SmoothingLength', 0, snapdir, snap_num, units_to_physical=units_to_physical)
        #gizmoSFR = load_from_snapshot.load_from_snapshot('StarFormationRate', 0, snapdir, snap_num, units_to_physical=units_to_physical)
        ages = np.empty(len(coords))
        ages.fill(-1)
        print ("Loaded gas data for snapshot {num}".format(num=snap_num))
        #return coords, masses, vels, ages, pID_array
    elif ptype == 1:
        print ("Loaded DM data for snapshot {num}".format(num=snap_num))
        #return coords, masses, vels, pID_array
    elif ptype == 2:
        print ("Loaded collisionless data for snapshot {num}".format(num=snap_num))
        #return coords, masses, vels, pID_array
    elif ptype == 4:
        sfts = load_from_snapshot.load_from_snapshot('StellarFormationTime', 4, snapdir, snap_num, units_to_physical=units_to_physical)
        print ("Loaded star data for snapshot {num}".format(num=snap_num))
        f = h5py.File(params.path+'snapdir_{num}/snapshot_{num}.0.hdf5'.format(num=snap_num), 'r')
        hubble_constant = f['Header'].attrs['HubbleParam']
        omega_matter = f['Header'].attrs['Omega0']
        omega_lambda = f['Header'].attrs['OmegaLambda']
        current_time = f['Header'].attrs['Time']
        f.close()
        co = Cosmology(hubble_constant=hubble_constant, \
                        omega_matter=omega_matter, omega_lambda=omega_lambda)
        #scales = sfts
        ages = np.array(co.t_from_a(current_time).in_units('Myr') - co.t_from_a(sfts).in_units('Myr'))
    else:
        print ("Invalid particle type...", ptype)
    
    return coords, masses, vels, ages, pID_array


def create_hdf5_file(snap_num, com_coords, com_vels, params, ic_path):
    """
    This is a function to create an hdf5 file with the COM coordinates and velocities of the SMBH particle.
    Inputs:
        snap_num: the snapshot number
        com_coords: the coordinates of the center of mass of the SMBH particle
        com_vels: the velocities of the center of mass of the SMBH
    
    Outputs:
        None
    """
    parts = 4
    header_data_dict_list = []
    gas_data_dict_list = []
    star_data_dict_list = []
    dm_data_dict_list = []
    collisionless_data_dict_list = []

    for i in range(0,parts):
        print ('Reading file {i} of {parts}....'.format(i=i, parts=parts))
        file_name = params.path+'snapdir_{snap_num}/snapshot_{snap_num}.{part}.hdf5'.format(snap_num=snap_num, part=i)
        f = h5py.File(file_name, 'r')
        header_data_dict = {}
        gas_data_dict = {}
        star_data_dict = {}
        dm_data_dict = {}
        collisionless_data_dict = {}
        for key in f.keys():
            #print(key)
            if key == 'Header':
                for key2 in f[key].attrs.keys():
                    #print (key2)
                    header_data_dict[key2] = f[key].attrs[key2]
            
            if key == 'PartType0':
                for key2 in f[key].keys():
                    gas_data_dict[key2] = np.array(f[key][key2])

            if key == 'PartType1':
                for key2 in f[key].keys():
                    dm_data_dict[key2] = np.array(f[key][key2])
            
            if key == 'PartType2':
                for key2 in f[key].keys():
                    collisionless_data_dict[key2] = np.array(f[key][key2])

            if key == 'PartType4':
                for key2 in f[key].keys():
                    star_data_dict[key2] = np.array(f[key][key2])
            
        header_data_dict_list.append(header_data_dict)
        gas_data_dict_list.append(gas_data_dict)
        star_data_dict_list.append(star_data_dict)
        dm_data_dict_list.append(dm_data_dict)
        collisionless_data_dict_list.append(collisionless_data_dict)
        f.close()

    
    # Now we can create the new hdf5 file with the SMBH particle (PartType3) added
    print ('Writing to file now ....')

    if not os.path.exists(ic_path):
        os.makedirs(ic_path)

    file_name = ic_path+'snapshot_{snap_num}.hdf5'.format(snap_num=snap_num)
    f = h5py.File(file_name, 'w')
    header = f.create_group('Header')
    for key in header_data_dict_list[0].keys():
        if key=='NumPart_ThisFile':
            arr = header_data_dict_list[0]['NumPart_ThisFile'] + \
                    header_data_dict_list[1]['NumPart_ThisFile'] + \
                        header_data_dict_list[2]['NumPart_ThisFile'] + \
                            header_data_dict_list[3]['NumPart_ThisFile']
            arr[3] = 1
            header.attrs.create(key, arr)
        else:
            header.attrs.create(key, header_data_dict_list[0][key])

    part0 = f.create_group('PartType0')
    for key in gas_data_dict_list[0].keys():
        if key == 'Coordinates' or key == 'Velocities':
            arr = np.vstack((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
        else:
            arr = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
        part0.create_dataset(key, data=arr)

    part1 = f.create_group('PartType1')
    for key in dm_data_dict_list[0].keys():
        if key == 'Coordinates' or key == 'Velocities':
            arr = np.vstack((dm_data_dict_list[0][key], dm_data_dict_list[1][key], dm_data_dict_list[2][key], dm_data_dict_list[3][key]))
        else:
            arr = np.concatenate((dm_data_dict_list[0][key], dm_data_dict_list[1][key], dm_data_dict_list[2][key], dm_data_dict_list[3][key]))
        part1.create_dataset(key, data=arr)
    
    part2 = f.create_group('PartType2')
    for key in collisionless_data_dict_list[0].keys():
        if key == 'Coordinates' or key == 'Velocities':
            arr = np.vstack((collisionless_data_dict_list[0][key], collisionless_data_dict_list[1][key], collisionless_data_dict_list[2][key], collisionless_data_dict_list[3][key]))
        else:
            arr = np.concatenate((collisionless_data_dict_list[0][key], collisionless_data_dict_list[1][key], collisionless_data_dict_list[2][key], collisionless_data_dict_list[3][key]))
        part2.create_dataset(key, data=arr)

    # Find max particle ID for all particle types in the snapshot
    max_pID = 0
    for i in range(0, parts):
        pID = np.max(gas_data_dict_list[i]['ParticleIDs'])
        if pID > max_pID:
            max_pID = pID
        pID = np.max(dm_data_dict_list[i]['ParticleIDs'])
        if pID > max_pID:
            max_pID = pID
        pID = np.max(collisionless_data_dict_list[i]['ParticleIDs'])
        if pID > max_pID:
            max_pID = pID
        pID = np.max(star_data_dict_list[i]['ParticleIDs'])
        if pID > max_pID:
            max_pID = pID


    part3 = f.create_group('PartType3')
    part3.create_dataset('Coordinates', data=com_coords)
    part3.create_dataset('Masses', data=np.array([1e-13]))
    part3.create_dataset('Velocities', data=com_vels)
    part3.create_dataset('ParticleIDs', data=np.array([max_pID+1]))
    

    part4 = f.create_group('PartType4')
    for key in star_data_dict_list[0].keys():
        if key == 'Coordinates' or key == 'Velocities':
            arr = np.vstack((star_data_dict_list[0][key], star_data_dict_list[1][key], star_data_dict_list[2][key], star_data_dict_list[3][key]))
        else:
            arr = np.concatenate((star_data_dict_list[0][key], star_data_dict_list[1][key], star_data_dict_list[2][key], star_data_dict_list[3][key]))
        part4.create_dataset(key, data=arr)

    f.close()


def get_close_particles(refine_pos_snap, params, refine_coords, dist_cut_off, follow_particle_types):
    """
    This function gets the particles that are close to the custom position of the refinement particle
    Inputs:
        refine_pos_snap: the snapshot number corresponding to the custom position of the refinement particle
        params: the Params object
        dist_cut_off: the distance cut off for gas particles to be used in finding COM velocity (in kpc)
    
    Outputs:
        close_coords: the coordinates of the particles that are close to the custom position
        close_vels: the velocities of the particles that are close to the custom position
        close_masses: the masses of the particles that are close to the custom position
        close_pID_array: the particle ID array of the particles that are close to the custom position
    """

    print ("Getting close particles...")
    #gas_coords, gas_masses, gas_vels, gas_ages, gas_pID_array = load_snapshot_data(params, refine_pos_snap, 0, units_to_physical=False)
    #star_coords, star_masses, star_vels, star_ages, star_pID_array = load_snapshot_data(params, refine_pos_snap, 4, units_to_physical=False)
    hubble_constant = 0.70124427
    ##final_refine_coords = np.array([41845.905, 44175.435, 46314.650])*hubble_constant
    
    count = 0
    for ptype in follow_particle_types:
        print ("Getting data for particle type...", ptype)
        coords, masses, vels, ages, pID_array = load_snapshot_data(params, refine_pos_snap, ptype, units_to_physical=False)

        print ("Calculating distances from", refine_coords, "for particle type...", ptype)
        dist_from_refine_coords = np.linalg.norm(coords - refine_coords, axis=1)
        inds = np.where(dist_from_refine_coords<dist_cut_off*hubble_constant)
        close_coords = coords[inds]
        close_vels = vels[inds]
        close_masses = masses[inds]
        close_pID_array = pID_array[inds]
        if count == 0:
            print ("Setting data...", len(close_coords))
            final_coords = close_coords
            final_vels = close_vels
            final_masses = close_masses
            final_pID_array = close_pID_array

        if count > 0:
            print ("Appending data...", len(close_coords))
            final_coords = np.vstack((final_coords, close_coords))
            final_vels = np.vstack((final_vels, close_vels))
            final_masses = np.concatenate((final_masses, close_masses))
            final_pID_array = np.vstack((final_pID_array, close_pID_array))

        count += 1
    
    print ("Final data shape...", final_coords.shape, final_vels.shape, final_masses.shape, final_pID_array.shape)
    return final_coords, final_vels, final_masses, final_pID_array


def get_COM_coords_vels(start_snap, params, tracked_pID_array, follow_particle_types):
    """
    This function calculates the center of mass coordinates 
    for a set of particles defined at some snapshot

    Inputs:
        snap_num: the snapshot number
        params: the Params object
        pID_array: the pID array of the set of particles 
                    to calculate the COM for

    Outputs:
        com_coords: the center of mass coordinates
        com_vels: the center of mass velocities
        cloud_coords: the coordinates of the cloud
    """
    print ("Getting COM coords and vels...")
    # Load the snapshot data
    count = 0
    for ptype in follow_particle_types:
        print ("Getting data for particle type...", ptype)
        coords, masses, vels, ages, pID_array = load_snapshot_data(params, refine_pos_snap, ptype, units_to_physical=False)
        if count == 0:
            final_coords = coords
            final_vels = vels
            final_masses = masses
            final_pID_array = pID_array

        if count > 0:
            print ("Appending data...", len(coords))
            final_coords = np.vstack((final_coords, coords))
            final_vels = np.vstack((final_vels, vels))
            final_masses = np.concatenate((final_masses, masses))
            final_pID_array = np.vstack((final_pID_array, pID_array))

        count += 1

    #coords, masses, vels, _, pID_array = load_snapshot_data(params, start_snap, 0, units_to_physical=False)
    
    # Find the particles in the cloud that are in the tracked cloud
    tracked_inds = np.where(np.isin(final_pID_array, tracked_pID_array).all(axis=1))[0]
    print ("Tracked particle number:", len(tracked_inds))
    
    tracked_coords = final_coords[tracked_inds]
    tracked_masses = final_masses[tracked_inds]
    tracked_vels = final_vels[tracked_inds]
    
    # Find the median coordinates of the tracked particles
    median_coords = np.median(tracked_coords, axis=0)

    # Find the particles that are within a certain distance of the median coordinates
    dist_from_median = np.linalg.norm(tracked_coords - median_coords, axis=1)
    inds = np.where(dist_from_median<4)
    tracked_coords = tracked_coords[inds]
    tracked_masses = tracked_masses[inds]
    tracked_vels = tracked_vels[inds]

    print ("Final tracked particle number:", len(tracked_coords))

    # Calculate the center of mass coordinates
    com_coords = np.sum(tracked_coords*tracked_masses[:, np.newaxis], axis=0)/np.sum(tracked_masses)

    # Calculate the center of mass velocities
    com_vels = np.sum(tracked_vels*tracked_masses[:, np.newaxis], axis=0)/np.sum(tracked_masses)

    print ("COM coords =", com_coords)
    print ("COM vels =", com_vels)
    return com_coords, com_vels


if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    ic_path = args['--ic_path']
    starting_snap = int(args['--starting_snap'])
    refine_pos_snap = int(args['--refine_pos_snap'])
    dist_cut_off = float(args['--dist_cut_off'])
    follow_particle_types = convert_to_array(args['--follow_particle_types'], dtype=np.int32)
    final_refine_coords = convert_to_array(args['--custom_refine_pos'], dtype=np.float64)
    hubble_constant = 0.70124427
    final_refine_coords = final_refine_coords*hubble_constant
        

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
    #sim = "m12i_final_fb_7k"
    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
    filename = path+sub_dir+filename_prefix+str(start_snap)+"_"+str(last_snap)+"names"+".txt"

    params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                    snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                    dat_file_header_size, star_data_sub_dir, cph_sub_dir,\
                    image_path, image_filename_prefix,\
                    image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)

    print ("Path = ", params.path)

    _, _, _, final_pID_array = get_close_particles(refine_pos_snap, params, final_refine_coords, dist_cut_off, follow_particle_types)

    #tracked_cloud_pID_array = get_tracked_cloud_pIDs(chain, params)

    #file_snap_num = chain.snap_nums[0]
    file_snap_num = starting_snap
    com_coords, com_vels = get_COM_coords_vels(file_snap_num, params, final_pID_array, follow_particle_types)

    ic_path = ic_path #+'/'#+chain.search_key+'/'
    if not os.path.exists(ic_path):
        os.makedirs(ic_path)
    
    create_hdf5_file(file_snap_num, com_coords, com_vels, params, ic_path)
    print("HDF5 file created successfully")



