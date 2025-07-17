#!/usr/bin/env python
"""
create_fire_ic_refine_tags: "Create initial conditions file for FIRE+STARFORGE simulation 
                            with a refinement tags on particles near a custom position.
                            The custom position is specified by the user. The script will
                            then place tags on particles within a certain distance of the custom
                            position.
                            One can specify if the particles should follow stars or gas particles while
                            deciding the original position for the refinement particle.
                            The script will then create an hdf5 file with the ICs."

Usage: create_fire_ic_refine_tags.py [options]

# e.g. python create_fire_ic_custom_pos.py --path=../../../FIRE-2/m12i_final/ --starting_snap=610 
# --custom_refine_pos=41845.905,44175.435,46314.650 --refine_pos_snap=615 
# --ic_path=Cloud0040Snap610_gas_dist0-03/ --dist_cut_off=0.03 --follow_particle_types=0


#create_fire_ic_custom_pos.py --path=../../../FIRE-2/m12i_final/ --starting_snap=610 --refine_pos=41845.905,44175.435,46314.650 --refine_pos_snap=615 --ic_path=Cloud0040Snap610_refineflag_gas_star_dist0-03_all_fixed --dist_cut_off=0.03 --follow_particle_types=0

Options:
    -h, --help                                          Show this screen
    --snapdir=<snapdir>                                 Are snapshots in a snapdir directory? [default: True]
    --path=<path>                                       Path to the simulation directory [default: ./]
    --sim=<sim>                                         Simulation name [default: m12i_final_fb_7k]
    --ic_path=<ic_path>                                 Path to save the IC file [default: ./] 
    --ic_file_name=<ic_file_name>                       Name of the IC file to be created [default: ic_refinement_tags]
    --snap_num=<snap_num>                               Snapshot number to create the IC file for [default: 610]
    --load_file_path=<load_file_path>                   Path to the file containing pIDs to tag [default: ./]
    --load_file_name=<load_file_name>                   Name of the file containing pIDs to tag [default: pIDs_to_tag.npy]
    --file_parts=<file_parts>                           Number of parts in the snapshot file [default: 1]
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



def load_pIDs_to_tag_from_file(params, load_file_path=None, load_file_name=None):

    if load_file_path is None:
        load_file_path = f"{params.path}"+f"ic_data/n{params.nmin}_alpha{params.vir}/" 
    else:
        if load_file_name is None:
            raise ValueError("Please provide a valid load_file_name.")
        load_file_path += load_file_name
    #elif load_file_name is not None:

    tracked_pID_array = np.load(load_file_path)
    return tracked_pID_array




def create_hdf5_file_flag_based_refinement(snap_num, params, ic_path, ic_file_name, load_file_path=None, load_file_name=None, file_parts=4, snapdir_flag=False):
    """
    This is a function to create an hdf5 file with the tags on particles which are to be used for calculating the COM of the refinement region.
    Inputs:
        snap_num: the snapshot number
    
    Outputs:
        None
    """
    #parts = file_parts
    header_data_dict_list = []
    gas_data_dict_list = []
    star_data_dict_list = []
    dm_data_dict_list = []
    collisionless_data_dict_list = []

    if snapdir_flag:
        print ('Reading files from snapdir...')
        snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
    else:
        snapdir = params.path

    for i in range(0,file_parts):
        print ('Reading file {k} of {parts}....'.format(k=i+1, parts=file_parts))
        if file_parts == 1:
            file_name = snapdir+'/snapshot_{snap_num}.hdf5'.format(snap_num=snap_num)
        else:
            file_name = snapdir+'/snapshot_{snap_num}.{part}.hdf5'.format(snap_num=snap_num, part=i)

        f = h5py.File(file_name, 'r')
        header_data_dict = {}
        gas_data_dict = {}
        star_data_dict = {}
        dm_data_dict = {}
        collisionless_data_dict = {}
        for key in f.keys():
            #print(key)
            print ('-------------------x---------------x---------------x-------------------')
            print ('Reading key:', key)
            if key == 'Header':
                for key2 in f[key].attrs.keys():
                    #print (key2)
                    header_data_dict[key2] = f[key].attrs[key2]
            
            if key == 'PartType0':
                for key2 in f[key].keys():
                    print (f'Reading {key2} in {key}...')
                    gas_data_dict[key2] = np.array(f[key][key2])

            if key == 'PartType1':
                for key2 in f[key].keys():
                    print (f'Reading {key2} in {key}...')
                    dm_data_dict[key2] = np.array(f[key][key2])
            
            if key == 'PartType2':
                for key2 in f[key].keys():
                    print (f'Reading {key2} in {key}...')
                    collisionless_data_dict[key2] = np.array(f[key][key2])

            if key == 'PartType4':
                for key2 in f[key].keys():
                    print (f'Reading {key2} in {key}...')
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

    file_name = ic_path+ic_file_name+".hdf5"
    #'snapshot_{snap_num}.hdf5'.format(snap_num=snap_num)
    f = h5py.File(file_name, 'w')
    header = f.create_group('Header')
    for key in header_data_dict_list[0].keys():
        if key=='NumPart_ThisFile':
            if file_parts == 1:
                arr = header_data_dict_list[0]['NumPart_ThisFile']
            else:
                arr = header_data_dict_list[0]['NumPart_ThisFile'] + \
                        header_data_dict_list[1]['NumPart_ThisFile'] + \
                            header_data_dict_list[2]['NumPart_ThisFile'] + \
                                header_data_dict_list[3]['NumPart_ThisFile']
            #arr[3] = 1
            header.attrs.create(key, arr)
        else:
            header.attrs.create(key, header_data_dict_list[0][key])


    print ("Writing PartType0 to file...")

    part0 = f.create_group('PartType0')
    for key in gas_data_dict_list[0].keys():
        if gas_data_dict_list[0][key].ndim>1:
            if file_parts == 1:
                arr = gas_data_dict_list[0][key]
            else:
                arr = np.vstack((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
        else:
            if file_parts == 1:
                arr = gas_data_dict_list[0][key]
            else:
                arr = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
        part0.create_dataset(key, data=arr)
    

    print ("Writing PartType1 to file...")
    part1 = f.create_group('PartType1')
    for key in dm_data_dict_list[0].keys():
        if dm_data_dict_list[0][key].ndim>1:
            if file_parts == 1:
                arr = dm_data_dict_list[0][key]
            else:
                arr = np.vstack((dm_data_dict_list[0][key], dm_data_dict_list[1][key], dm_data_dict_list[2][key], dm_data_dict_list[3][key]))
        else:
            if file_parts == 1:
                arr = dm_data_dict_list[0][key]
            else:
                arr = np.concatenate((dm_data_dict_list[0][key], dm_data_dict_list[1][key], dm_data_dict_list[2][key], dm_data_dict_list[3][key]))
        part1.create_dataset(key, data=arr)
    if file_parts == 1:
        pIDs = dm_data_dict_list[0]['ParticleIDs']
    else:
        pIDs = np.concatenate((dm_data_dict_list[0][key], dm_data_dict_list[1][key], dm_data_dict_list[2][key], dm_data_dict_list[3][key]))
    refine_flag_array = np.zeros(len(pIDs))
    refine_flag_array = refine_flag_array.astype('int32')
    part1.create_dataset("RefinementFlag", data=refine_flag_array)

    
    print ("Writing PartType2 to file...")
    part2 = f.create_group('PartType2')
    for key in collisionless_data_dict_list[0].keys():
        if collisionless_data_dict_list[0][key].ndim>1:
            if file_parts == 1:
                arr = collisionless_data_dict_list[0][key]
            else:
                arr = np.vstack((collisionless_data_dict_list[0][key], collisionless_data_dict_list[1][key], collisionless_data_dict_list[2][key], collisionless_data_dict_list[3][key]))
        else:
            if file_parts == 1:
                arr = collisionless_data_dict_list[0][key]
            else:
                arr = np.concatenate((collisionless_data_dict_list[0][key], collisionless_data_dict_list[1][key], collisionless_data_dict_list[2][key], collisionless_data_dict_list[3][key]))
        part2.create_dataset(key, data=arr)
    if file_parts == 1:
        pIDs = collisionless_data_dict_list[0]['ParticleIDs']
    else:
        pIDs = np.concatenate((collisionless_data_dict_list[0][key], collisionless_data_dict_list[1][key], collisionless_data_dict_list[2][key], collisionless_data_dict_list[3][key]))
    refine_flag_array = np.zeros(len(pIDs))
    refine_flag_array = refine_flag_array.astype('int32')
    part2.create_dataset("RefinementFlag", data=refine_flag_array)

    
    print ("Writing PartType4 to file...")
    part4 = f.create_group('PartType4')
    for key in star_data_dict_list[0].keys():
        if star_data_dict_list[0][key].ndim>1:
            if file_parts == 1:
                arr = star_data_dict_list[0][key]
            else:
                arr = np.vstack((star_data_dict_list[0][key], star_data_dict_list[1][key], star_data_dict_list[2][key], star_data_dict_list[3][key]))
        else:
            if file_parts == 1:
                arr = star_data_dict_list[0][key]
            else:
                arr = np.concatenate((star_data_dict_list[0][key], star_data_dict_list[1][key], star_data_dict_list[2][key], star_data_dict_list[3][key]))
        part4.create_dataset(key, data=arr)
    if file_parts == 1:
        pIDs = star_data_dict_list[0]['ParticleIDs']
    else:
        pIDs = np.concatenate((star_data_dict_list[0][key], star_data_dict_list[1][key], star_data_dict_list[2][key], star_data_dict_list[3][key]))
    refine_flag_array = np.zeros(len(pIDs))
    refine_flag_array = refine_flag_array.astype('int32')
    part4.create_dataset("RefinementFlag", data=refine_flag_array)






    # Add the refinement flag to the gas particles
    # The refinement flag array is a 1D array with the same length as the number of gas particles, 1 for particles to be used in the COM calculation and 0 otherwise
    #_, _, pID_array_for_tagging = load_pIDs_to_tag_from_file(params, snap_num, refine_pos_snap, final_refine_coords, dist_cut_off, follow_particle_types, units_to_physical=False)

    print ("Loading pIDs to tag from file...")
    pID_array_for_tagging = load_pIDs_to_tag_from_file(params, load_file_path=load_file_path, load_file_name=load_file_name)
    



    # Now whereever the pID_array_for_tagging matches the gas particle pID, we set the refinement flag to 1
    key = 'ParticleIDs'
    if file_parts == 1:
        pIDs = gas_data_dict_list[0][key]
    else:
        pIDs = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
    key = 'ParticleIDGenerationNumber'
    if file_parts == 1:
        pID_gen_nums = gas_data_dict_list[0][key]
    else:
        pID_gen_nums = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
    key = 'ParticleChildIDsNumber'
    if file_parts == 1:
        pID_child_nums = gas_data_dict_list[0][key]
    else:
        pID_child_nums = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
    
    pID_array = np.column_stack((pIDs, pID_gen_nums, pID_child_nums))
    


    print ("Preparing refinement flag array...")

    # Now check where pID_array_for_tagging matches the pID array fully and set those indices to 1
    refinement_flag_array = np.zeros(len(pIDs))
    inds = np.where((pID_array==pID_array_for_tagging[:,None]).all(-1))[1]
    refinement_flag_array[inds]=1
    #refinement_flag_array = refinement_flag_array.astype('uint32')
    refinement_flag_array = refinement_flag_array.astype('int32')
    part0.create_dataset("RefinementFlag", data=refinement_flag_array)
    
    key = "Coordinates"
    if file_parts == 1:
        coords_arr = gas_data_dict_list[0][key]
    else:
        coords_arr = np.vstack((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
    key = "Masses"        
    if file_parts == 1:
        masses_arr = gas_data_dict_list[0][key]
    else:
        masses_arr = np.concatenate((gas_data_dict_list[0][key], gas_data_dict_list[1][key], gas_data_dict_list[2][key], gas_data_dict_list[3][key]))
    
    com_coords_x = np.take(coords_arr[:,0], inds)
    com_coords_y = np.take(coords_arr[:,1], inds)
    com_coords_z = np.take(coords_arr[:,2], inds)
    com_masses = np.take(masses_arr, inds)
    refinement_center_x = np.sum(com_coords_x*com_masses)/np.sum(com_masses)
    refinement_center_y = np.sum(com_coords_y*com_masses)/np.sum(com_masses)
    refinement_center_z = np.sum(com_coords_z*com_masses)/np.sum(com_masses)
    refinement_center = np.array([refinement_center_x, refinement_center_y, refinement_center_z]).T
    print ("Refinement Center:", refinement_center)
    
    header.attrs.create("RefinementRegionCenter", refinement_center)




    
    f.close()











if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    sim = args['--sim']
    snapdir = args['--snapdir']
    snap_num = int(args['--snap_num'])
    ic_path = args['--ic_path']
    ic_file_name = args['--ic_file_name']
    load_file_path = args['--load_file_path']
    load_file_name = args['--load_file_name']
    #flag_based_refinement = convert_to_bool(args['--flag_based_refinement'])
    snapdir = convert_to_bool(args['--snapdir'])
    file_parts = int(args['--file_parts'])

    ## Some bookkeeping
    path = "/mnt/raid-project/murray/khullar/FIRE-3/"
    start_snap = 500
    last_snap = 900
    linked_filename_prefix = "Linked_Clouds_"
    cloud_num_digits = 4
    snapshot_num_digits = 4
    cloud_prefix = "Cloud"
    hdf5_file_prefix = 'Clouds_'
    sim = 'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5'
    age_cut = 5
    dat_file_header_size = 11
    snapshot_prefix = "Snap"
    star_data_sub_dir = "StarData/"
    gas_data_sub_dir = "GasData/"
    cph_sub_dir = "CloudPhinderData/"
    r_gal = 25
    h = 0.4
    nmin = 1
    vir = 10

    threshold = '0.5'
    frac_thresh = 'thresh' + threshold
    sub_dir = f"CloudPhinderData/n{nmin}_alpha{vir}/"
    vels_sub_dir = 'vel_sub/'
    gal_quants_sub_dir = 'gal_quants/'

    params = Params(path=path, sub_dir=sub_dir, start_snap=start_snap, last_snap=last_snap,
                    filename_prefix=linked_filename_prefix, cloud_prefix=cloud_prefix,
                    hdf5_file_prefix=hdf5_file_prefix, frac_thresh=frac_thresh, sim=sim,
                    r_gal=r_gal, h=h, gal_quants_sub_dir=gal_quants_sub_dir, vels_sub_dir=vels_sub_dir,
                    phinder_sub_dir=cph_sub_dir, age_cut=age_cut,
                    dat_file_header_size=dat_file_header_size, nmin=nmin, vir=vir,
                    cloud_num_digits=cloud_num_digits, snapshot_num_digits=snapshot_num_digits, verbose=False)

    params.linked_filename = f"{params.path}{params.sub_dir}{params.filename_prefix}n{params.nmin}"+\
                            f"_alpha{params.vir}_{params.frac_thresh}_{params.start_snap}_{params.last_snap}_names.txt"

    print ("Path = ", params.path)

    #print ("Flag Based Refinement mode...")
    #get_pIDs_to_tag(params, file_snap_num, refine_pos_snap, final_refine_coords, dist_cut_off, follow_particle_types, units_to_physical=False)
    
    create_hdf5_file_flag_based_refinement(snap_num, params, ic_path, ic_file_name, load_file_path, load_file_name, file_parts=file_parts, snapdir_flag=snapdir)


    print("HDF5 file created successfully")



