"""
This file contains functions to get individual cloud quantities, as well as entire population quantities.

Author: Shivan Khullar
Date: April 2024
"""


import sys
sys.path.append('/src/generic_utils/')
from fire_utils import *
from overall_cloud_props import *





########################################################################################
########################### Getting cloud quantities   #################################
########################################################################################


def get_quick_cloud_info(path, snapnum, nmin, vir, cloud_num):
    """
    This is a function to get the quantities of a cloud from the bound file.
    
    Inputs:
        path: the path to the data
        snapnum: the snapshot number
        nmin: the minimum number of particles in a cloud
        vir: the virial parameter
        cloud_num: the cloud number
    
    Outputs:
        cloud_total_mass: the total mass of the cloud
        cloud_centre: the centre of the cloud
        cloud_reff: the effective radius of the cloud
    """
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
    
    cloud_centre = np.array([cloud_centre_x, cloud_centre_y, cloud_centre_z])
    return cloud_total_mass, cloud_centre, cloud_reff



def get_cloud_quants_hdf5(cloud_num, snap_num, params):
    """
    This is a function to get the quantities of a cloud from the hdf5 file.
    Inputs:
        cloud_num: the cloud number
        snap_num: the snapshot number
        params: the parameters (see src/fire_utils.py)

    Outputs:
        dens: the density of the cloud
        vels: the velocities of the cloud
        coords: the coordinates of the cloud
        masses: the gas masses of the cloud
        hsml: the smoothing lengths of the cloud
        cs: the sound speeds of the cloud

    """
    file_name = params.path+params.cph_sub_dir+params.hdf5_file_prefix+str(snap_num)+\
                '_n{nmin}_alpha{vir}'.format(nmin=params.nmin, vir=params.vir)+'.hdf5'
    f = h5py.File(file_name, 'r')
    cloud_name = get_cloud_name(cloud_num, params)
    g = f[cloud_name]
    
    h = g['PartType0']
    dens = np.array(h['Density'])
    vels = np.array(h['Velocities'])
    coords = np.array(h['Coordinates'])
    masses = np.array(h['Masses'])

    hsml = np.array(h['SmoothingLength'])
    #phi = np.array(h['Potential'])
    #gizmo_sfrs = np.array(h['StarFormationRate'])
    int_energy = np.array(h['InternalEnergy'])
    metal = np.array(h['Metallicity'])
    n_elec = np.array(h['ElectronAbundance'])
    f_neutral = np.zeros(0)
    f_molec = np.zeros(0)
    z_tot = metal[:, 0]
    z_he = metal[:, 1]
    
   
    value='Temp'
    temps = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key=value)
    
    value='Weight'
    weights = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key=value)
    
    m_p = 1.67e-24          # mass of proton (g)
    k_B = 1.38e-16          # Boltzmann constant (erg/K)
    c_s = np.sqrt(k_B*temps/(weights*m_p))/(1e5)
    #cs_mw = np.sqrt(np.sum(masses*c_s**2)/np.sum(masses))

    return dens, vels, coords, masses, hsml, c_s, temps




def get_cloud_box(cloud_num, snap_num, params, cloud_reff_factor=1.5, cloud_box=None):
    """ 
    Get quantities of the cloud within a box around the cloud when snapshot arrays are available.
    Doing this is typically useful when say we have to make a movie of just one cloud across different snapshots
    and want to track the box around the cloud.

    Parameters
    ----------
    cloud_num : int
        The cloud number
    snap_num : int
        The snapshot number
    params : object
        The parameters
    cloud_reff_factor : float
        The factor by which to multiply the effective radius of the cloud
    cloud_box : CloudBox object
        The box around the cloud

    Returns
    -------
    final_gas_coords : array_like
        The coordinates of the gas particles within the box
    final_gas_masses : array_like
        The masses of the gas particles within the box
    final_gas_smoothing_lengths : array_like
        The smoothing lengths of the gas particles within the box
    final_gas_densities : array_like
        The densities of the gas particles within the box
    final_gas_temps : array_like
        The temperatures of the gas particles within the box
    final_gas_vels : array_like
        The velocities of the gas particles within the box
    cloud_box : CloudBox object 
    """
    _, cloud_centre, cloud_reff = get_cloud_info(params.path, snap_num, \
                                                            params.nmin, params.vir, cloud_num)
    
    
    # Define the box around the cloud
    if cloud_box is None:
        cloud_box = CloudBox()
        cloud_box.x_max = cloud_centre[0] + cloud_reff_factor*cloud_reff
        cloud_box.x_min = cloud_centre[0] - cloud_reff_factor*cloud_reff

        cloud_box.y_max = cloud_centre[1] + cloud_reff_factor*cloud_reff
        cloud_box.y_min = cloud_centre[1] - cloud_reff_factor*cloud_reff

        cloud_box.z_max = cloud_centre[2] + cloud_reff_factor*cloud_reff
        cloud_box.z_min = cloud_centre[2] - cloud_reff_factor*cloud_reff
    else:
        print ('Using the cloud box provided')

    print ('Loading particle data...')
    
    # Load the particle data
    dens = Load_FIRE_Data_Arr('gas', 'dens', snap_num, params)
    coords = Load_FIRE_Data_Arr('gas', 'coords', snap_num, params)
    masses = Load_FIRE_Data_Arr('gas', 'masses', snap_num, params)
    temps = Load_FIRE_Data_Arr('gas', 'temps', snap_num, params)
    vels = Load_FIRE_Data_Arr('gas', 'vels', snap_num, params)
    hsml = Load_FIRE_Data_Arr('gas', 'hsml', snap_num, params)

    # Select the particles within the box
    inds_x = np.where((coords[:,0]>cloud_box.x_min)&(coords[:,0]<cloud_box.x_max))[0]
    inds_y = np.where((coords[:,1]>cloud_box.y_min)&(coords[:,1]<cloud_box.y_max))[0]
    inds_z = np.where((coords[:,2]>cloud_box.z_min)&(coords[:,2]<cloud_box.z_max))[0]


    final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)

    coords_x = np.take(coords[:,0], final_inds)
    coords_y = np.take(coords[:,1], final_inds)
    coords_z = np.take(coords[:,2], final_inds)
    final_gas_coords = np.array([coords_x, coords_y, coords_z]).T

    final_gas_masses = np.take(masses, final_inds)
    final_gas_smoothing_lengths = np.take(hsml, final_inds)
    final_gas_densities = np.take(dens, final_inds)
    final_gas_temps = np.take(temps, final_inds)
    vels_x = np.take(vels[:,0], final_inds)
    vels_y = np.take(vels[:,1], final_inds)
    vels_z = np.take(vels[:,2], final_inds)
    final_gas_vels = np.array([vels_x, vels_y, vels_z]).T

    return final_gas_coords, final_gas_masses, final_gas_smoothing_lengths, \
        final_gas_densities, final_gas_temps, final_gas_vels, cloud_box








class CloudChain():
    """
    Class to find out a cloud chain stemming from the first cloud.
    """
    def __init__(self, cloud_num, snap_num, params):
        file_name = params.path+params.sub_dir+params.filename_prefix+params.frac_thresh\
                    +"_"+str(params.start_snap)+"_"+str(params.last_snap)+"_names"+".txt"
        my_file = open(file_name, "r")

        content_list = my_file.readlines()
        cloud_list_names = []
        for i in range (0, len(content_list)-1):             #The last line is just \n.
            #if 
            names = str.split(content_list[i], ', ')
            if names[-1]=='\n':
                names = names[:-1]

            cloud_list_names.append(names)

        self.cloud_list = []
        self.cloud_nums = []
        self.snap_nums = []
        search_key = get_cloud_name(cloud_num, params)+'Snap'+str(snap_num)
        flag = 0
        for i in range(0, len(cloud_list_names)):
            if search_key in cloud_list_names[i]:
                print ('Search key', search_key, i)
                self.cloud_list = cloud_list_names[i]

                flag = 1
                break
        
        for cloud in self.cloud_list:
            self.cloud_nums.append(int(cloud.split('Snap')[0].split('Cloud')[1]))
            self.snap_nums.append(int(cloud.split('Snap')[1]))
        
        #if flag==0:
        #    search_key = get_cloud_name0(cloud_num, params)+'Snap'+str(snap_num)
        #    for i in range(0, len(cloud_list_names)):
        #        if search_key in cloud_list_names[i]:
        #            print ('Search key', search_key, i)
        #            self.cloud_list = cloud_list_names[i]
        #            flag = 1
        #            break
                    
        if flag==0:
            print ('Cloud not found :(')
            
