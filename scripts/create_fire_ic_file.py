#!/usr/bin/env python
"""
create_fire_ic_file: "Create initial conditions file for FIRE+STARFORGE simulation"

Usage: create_fire_ic_file.py [options]

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



def get_cloud_pIDs(cloud_num, snap_num, params):
    """
    This is a function to get the particle IDs of a cloud from the hdf5 file.
    Inputs:
        cloud_num: the cloud number
        snap_num: the snapshot number
        params: the parameters (see src/fire_utils.py)

    Outputs:
        pIDs: the particle IDs of the cloud

    """
    file_name = params.path+params.cph_sub_dir+params.hdf5_file_prefix+str(snap_num)+\
                '_n{nmin}_alpha{vir}'.format(nmin=params.nmin, vir=params.vir)+'.hdf5'
    f = h5py.File(file_name, 'r')
    cloud_name = get_cloud_name(cloud_num, params)
    g = f[cloud_name]
    h = g['PartType0']
    pIDs = np.array(h['ParticleIDs'])
    pID_gen_nums = np.array(h['ParticleIDGenerationNumber'])
    pID_child_nums = np.array(h['ParticleChildIDsNumber'])
    return pIDs, pID_gen_nums, pID_child_nums


def get_tracked_cloud_pIDs(chain, params):
    pIDs = []
    pID_gen_nums = []
    pID_child_nums = []
    for i in range(0, len(chain.cloud_list)):
        cloud_num = chain.cloud_nums[i]
        snapnum = chain.snap_nums[i]
        cloud_pIDs, cloud_pID_gen_nums, cloud_pID_child_nums = get_cloud_pIDs(cloud_num, snapnum, params)
        pIDs.append(cloud_pIDs)
        pID_gen_nums.append(cloud_pID_gen_nums)
        pID_child_nums.append(cloud_pID_child_nums)

    # Get rid of duplicates in this list
    pIDs = np.concatenate(pIDs)
    pID_gen_nums = np.concatenate(pID_gen_nums)
    pID_child_nums = np.concatenate(pID_child_nums)
    pID_array = np.column_stack((pIDs, pID_gen_nums, pID_child_nums))
    pID_array = np.unique(pID_array, axis=0)
    
    return pID_array



def get_cloud_info(path, snapnum, nmin, vir, cloud_num, params):
    path = path+'CloudPhinderData/n{nmin}_alpha{vir}/'.format(nmin=nmin, vir=vir)
    filename = 'bound_{snap_num}_n{nmin}_alpha{vir}.dat'.format(snap_num = snapnum, nmin = nmin, vir=vir)
    datContent = [i.strip().split() for i in open(path+filename).readlines()]
    
    
    #for i in range (0, len(datContent)):
    #    if (i<8):
    #        continue
    
    header = 8
    i = cloud_num
    cloud_total_mass = float(datContent[i+header][0])
    cloud_centre_x = float(datContent[i+header][1])
    cloud_centre_y = float(datContent[i+header][2])
    cloud_centre_z = float(datContent[i+header][3])
    cloud_reff = float(datContent[i+header][7])
    cloud_vir = float(datContent[i+header][10])
    
    cloud_centre = np.array([cloud_centre_x, cloud_centre_y, cloud_centre_z])
    gal_centre = Get_Galaxy_Centre(params, snapnum)
    dist = np.sqrt((cloud_centre_x-gal_centre[0])**2+(cloud_centre_y-gal_centre[1])**2+\
                    (cloud_centre_z-gal_centre[2])**2)
    
    return cloud_total_mass, cloud_centre, cloud_reff, cloud_vir, dist


def get_maximum_cloud_reff(chain, params):
    # Find the maximum cloud radius over all snapshots
    max_reff = 0
    for i in range(0, len(chain.cloud_list)):
        snap_num = chain.snap_nums[i]
        cloud_num = chain.cloud_nums[i]
        
        _, _, cloud_reff,_,_ = get_cloud_info(params.path, snap_num, \
                                                                params.nmin, params.vir, cloud_num, params)
        if i==0:
            init_cloud_reff = cloud_reff

        if cloud_reff > max_reff:
            max_reff = cloud_reff
            max_reff_cloud_num = cloud_num
            max_reff_snap_num = snap_num

    cloud_reff_factor = (max_reff/init_cloud_reff)*1.2

    return max_reff, max_reff_cloud_num, max_reff_snap_num, cloud_reff_factor


def get_COM_coords_vels(snap_num, params, tracked_cloud_pID_array):
    """
    This is a function to get the coordinates of the center of mass of a set of particles.
    Inputs:
        pIDs: the particle IDs of the particles
        snap_num: the snapshot number
        params: the parameters (see src/fire_utils.py)
    Outputs:
        coords: the coordinates of the center of mass of the particles
    """
    snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
    coords = load_from_snapshot.load_from_snapshot('Coordinates', 0, snapdir, snap_num, units_to_physical=False)
    masses = load_from_snapshot.load_from_snapshot('Masses', 0, snapdir, snap_num, units_to_physical=False)
    vels = load_from_snapshot.load_from_snapshot('Velocities', 0, snapdir, snap_num, units_to_physical=False)
    pIDs = load_from_snapshot.load_from_snapshot('ParticleIDs', 0, snapdir, snap_num, units_to_physical=False)
    pID_gen_nums = load_from_snapshot.load_from_snapshot('ParticleIDGenerationNumber', 0, snapdir, snap_num, units_to_physical=False)
    pID_child_nums = load_from_snapshot.load_from_snapshot('ParticleChildIDsNumber', 0, snapdir, snap_num, units_to_physical=False)
    pID_array = np.column_stack((pIDs, pID_gen_nums, pID_child_nums))
    # Get the indices of the particles we want -- basically the ones which are in the tracked_cloud_pID_array
    check = np.isin(pID_array, tracked_cloud_pID_array)
    indices = np.where(check.all(axis=1))[0]
    #indices = np.where(np.in1d(pID_array, tracked_cloud_pID_array).reshape(pID_array.shape))[0]
    coords = coords[indices]
    vels = vels[indices]
    masses = masses[indices]
    # We have to clean the data a bit, there are still particles far away from the cloud that have the same pID, gen_num, child_num.

    #Find the median x, y and z coordinates from coords and exclude particles that are more than 4 kpc away from the median.
    max_reff, max_reff_cloud_num, max_reff_snap_num, cloud_reff_factor = get_maximum_cloud_reff(chain, params)

    median_coords = np.median(coords, axis=0)
    distances = np.linalg.norm(coords - median_coords, axis=1)
    new_indices = np.where(distances < 2*max_reff)[0]
    coords = coords[new_indices]
    vels = vels[new_indices]
    masses = masses[new_indices]

    # Now we can calculate the center of mass
    COM_coords = np.average(coords, axis=0, weights=masses)
    COM_vels = np.average(vels, axis=0, weights=masses)

    return COM_coords, COM_vels, coords


def create_hdf5_file(snap_num, com_coords, com_vels, cloud_coords, params, ic_path):
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


    ## Assuming we know which cloud we're placing the SMBH particle in...
    cloud_num = 40
    snap_num = int(args['--snapnum'])
    chain = CloudChain(cloud_num, snap_num, params)    

    tracked_cloud_pID_array = get_tracked_cloud_pIDs(chain, params)

    file_snap_num = chain.snap_nums[0]
    com_coords, com_vels, cloud_coords = get_COM_coords_vels(file_snap_num, params, tracked_cloud_pID_array)

    ic_path = ic_path+chain.search_key+'/'
    if not os.path.exists(ic_path):
        os.makedirs(ic_path)
    
    create_hdf5_file(file_snap_num, com_coords, com_vels, cloud_coords, params, ic_path)
    print("HDF5 file created successfully")