"""
This file contains functions to get individual cloud quantities, as well as entire population quantities.

Author: Shivan Khullar
Date: April 2024
"""


#import sys
#sys.path.append('/src/generic_utils/')
from generic_utils.fire_utils import *
from cloud_utils.cloud_utils import *
import scipy


########################################################################################
########################### Getting cloud quantities   #################################
########################################################################################




def get_cloud_quants_hdf5(cloud_num, snap_num, params):
    """
    This is a function to get the quantities of a cloud from the hdf5 file.
    Inputs:
        cloud_num: the cloud number
        snap_num: the snapshot number
        params: the parameters (see src/fire_utils.py)

    Outputs:
        dens: the density of the particles in the cloud
        vels: the velocities of the particles in the cloud
        coords: the coordinates of the particles in the cloud
        masses: the gas masses of the particles in the cloud
        hsml: the smoothing lengths of the particles in the cloud
        cs: the sound speeds of the particles in the cloud
        temps: the temperatures of the particles in the cloud
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
    _, cloud_centre, cloud_reff, _, _ = get_cloud_quick_info(snap_num, \
                                                            params.nmin, params.vir, cloud_num, params)
    
    
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
    dens = load_fire_data_arr('gas', 'dens', snap_num, params)
    coords = load_fire_data_arr('gas', 'coords', snap_num, params)
    masses = load_fire_data_arr('gas', 'masses', snap_num, params)
    temps = load_fire_data_arr('gas', 'temps', snap_num, params)
    vels = load_fire_data_arr('gas', 'vels', snap_num, params)
    hsml = load_fire_data_arr('gas', 'hsml', snap_num, params)

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




def get_cloud_quants(cloud_num, snap_num, params, cloud_reff_factor=1.5, cloud_box=None, \
                     snap_data=None, project=False, center_wrt_galaxy=False, star_data=True):
    """ 
    Function to get the quantities of a cloud.

    Inputs:
        cloud_num: Cloud number.
        snap_num: Snapshot number.
        params: Parameters of the simulation.
        cloud_reff_factor: Factor to multiply the cloud effective radius to get the box size.
        cloud_box: If True, a box around the cloud is defined. If a dictionary, that is used as the box.
        snap_data: If not None, the snapshot data is provided.
        project: If True, the cloud is projected.
        center_wrt_galaxy: If True, the cloud is centered wrt the galaxy.
    
    Outputs:
        dens: Density of the cloud.
        vels: Velocities of the cloud.
        coords: Coordinates of the cloud.
        masses: Masses of the cloud.
        hsml: Smoothing lengths of the cloud.
        cs: Sound speeds of the cloud.
    """
    #if project==True:
    #    print ('This will only project if cloud_box=False ...')

    _, cloud_centre, cloud_reff, _, _ = get_cloud_quick_info(snap_num, \
                                                            params.nmin, params.vir, cloud_num, params)
    
    mode = 'box'
    # Define the box around the cloud
    if cloud_box is None or cloud_box == True:
        #cloud_box = CloudBox()
        cloud_box = dict.fromkeys(['x_max', 'x_min', 'y_max', 'y_min', 'z_max', 'z_min']) 
        cloud_box['x_max'] = cloud_centre[0] + cloud_reff_factor*cloud_reff
        cloud_box['x_min'] = cloud_centre[0] - cloud_reff_factor*cloud_reff

        cloud_box['y_max'] = cloud_centre[1] + cloud_reff_factor*cloud_reff
        cloud_box['y_min'] = cloud_centre[1] - cloud_reff_factor*cloud_reff

        cloud_box['z_max'] = cloud_centre[2] + cloud_reff_factor*cloud_reff
        cloud_box['z_min'] = cloud_centre[2] - cloud_reff_factor*cloud_reff
    # If cloud_box is a dictionary use that, else just return the cloud from get_cloud_quants_hdf5
    elif type(cloud_box) == dict:
        print ('Using the cloud box provided...')
    if cloud_box == False:
        mode = 'cloud'


    #print ('Loading particle data...')
    if mode == 'cloud':
        # Load the particle data
        dens, vels, coords, masses, hsml, cs, temps = get_cloud_quants_hdf5(cloud_num, snap_num, params)
        if project:
            print ("Projecting the cloud...")

            proj = get_galaxy_proj_matrix(params, snap_num)
            proj_gas_coords = []
            for i in range(0, len(gas_coords)):
                proj_gas_coords.append(np.matmul(proj, gas_coords[i]))
            proj_gas_coords = np.array(proj_gas_coords)

            if center_wrt_galaxy:
                print ('Centering wrt galaxy...')
                gal_centre = get_galaxy_centre(params, snap_num)    
                proj = get_galaxy_proj_matrix(params, snap_num)
                gal_centre_proj = np.matmul(proj, gal_centre)
                proj_gas_coords = proj_gas_coords - gal_centre_proj

            return dens, vels, proj_gas_coords, masses, hsml, cs
        
        if star_data:
            if snap_data is not None:
                star_coords = snap_data['star_coords']
                star_masses = snap_data['star_masses']
            else:
                star_coords = load_fire_data_arr('star', 'coords', snap_num, params)
                star_masses = load_fire_data_arr('star', 'masses', snap_num, params)
            
            dists = np.sqrt((star_coords[:,0]-cloud_centre[0])**2+\
                            (star_coords[:,1]-cloud_centre[1])**2+\
                            (star_coords[:,2]-cloud_centre[2])**2)
            star_inds = np.where(dists<=cloud_reff)[0]
            final_star_coords = star_coords[star_inds]
            final_star_masses = star_masses[star_inds]
            
            return dens, vels, coords, masses, hsml, cs, temps, \
                    final_star_coords, final_star_masses

        return dens, vels, coords, masses, hsml, cs, temps

    if mode == 'box':
        if snap_data is not None:
            # Select the particles within the box
            coords = snap_data['coords']
            masses = snap_data['masses']
            hsml = snap_data['hsml']
            dens = snap_data['dens']
            temps = snap_data['temps']
            vels = snap_data['vels']
            cs = snap_data['cs']
            int_energy = snap_data['int_energy']
            pIDs = snap_data['pIDs']
        else:
            # Load the snapshot data
            dens = load_fire_data_arr('gas', 'dens', snap_num, params)
            coords = load_fire_data_arr('gas', 'coords', snap_num, params)
            masses = load_fire_data_arr('gas', 'masses', snap_num, params)
            temps = load_fire_data_arr('gas', 'temps', snap_num, params)
            vels = load_fire_data_arr('gas', 'vels', snap_num, params)
            hsml = load_fire_data_arr('gas', 'hsml', snap_num, params)
            int_energy = load_fire_data('InternalEnergy', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            metal = load_fire_data('Metallicity', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            n_elec = load_fire_data('ElectronAbundance', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            f_neutral = np.zeros(0)
            f_molec = np.zeros(0)
            z_tot = metal[:, 0]
            z_he = metal[:, 1]

            value='Temp'
            if not temps:
                temps = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key=value)
            weights = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key='Weight')

            m_p = 1.67e-24          # mass of proton (g)
            k_B = 1.38e-16          # Boltzmann constant (erg/K)
            cs = np.sqrt(k_B*temps/(weights*m_p))/(1e5)  # in km/s


        inds_x = np.where((coords[:,0]>cloud_box['x_min'])&(coords[:,0]<cloud_box['x_max']))[0]
        inds_y = np.where((coords[:,1]>cloud_box['y_min'])&(coords[:,1]<cloud_box['y_max']))[0]
        inds_z = np.where((coords[:,2]>cloud_box['z_min'])&(coords[:,2]<cloud_box['z_max']))[0]
        final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)
    
        coords_x = np.take(coords[:,0], final_inds)
        coords_y = np.take(coords[:,1], final_inds)
        coords_z = np.take(coords[:,2], final_inds)
        final_gas_coords = np.array([coords_x, coords_y, coords_z]).T

        final_gas_masses = np.take(masses, final_inds)
        final_gas_smoothing_lengths = np.take(hsml, final_inds)
        final_gas_densities = np.take(dens, final_inds)
        final_gas_temps = np.take(temps, final_inds)
        final_gas_cs = np.take(cs, final_inds)
        final_gas_int_energy = np.take(int_energy, final_inds)
        final_temps = np.take(temps, final_inds)
        vels_x = np.take(vels[:,0], final_inds)
        vels_y = np.take(vels[:,1], final_inds)
        vels_z = np.take(vels[:,2], final_inds)
        final_gas_vels = np.array([vels_x, vels_y, vels_z]).T
        final_gas_pIDs = pIDs[final_inds]
        
        if star_data:
            if snap_data is not None:
                star_coords = snap_data['star_coords']
                star_masses = snap_data['star_masses']
                star_vels = snap_data['star_vels']
                star_pIDs = snap_data['star_pIDs']
                sfts = snap_data['sfts']
            else:
                star_coords = load_fire_data_arr('star', 'coords', snap_num, params)
                star_masses = load_fire_data_arr('star', 'masses', snap_num, params)
                star_vels = load_fire_data('Velocities', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
                star_pIDs = load_fire_data('ParticleIDs', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
                sfts = load_fire_data('StellarFormationTime', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)

            inds_x = np.where((star_coords[:,0]>cloud_box['x_min'])&(star_coords[:,0]<cloud_box['x_max']))[0]
            inds_y = np.where((star_coords[:,1]>cloud_box['y_min'])&(star_coords[:,1]<cloud_box['y_max']))[0]
            inds_z = np.where((star_coords[:,2]>cloud_box['z_min'])&(star_coords[:,2]<cloud_box['z_max']))[0]
            final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)

            final_star_coords = star_coords[final_inds]
            final_star_masses = star_masses[final_inds]
            vels_x = np.take(star_vels[:,0], final_inds)
            vels_y = np.take(star_vels[:,1], final_inds)
            vels_z = np.take(star_vels[:,2], final_inds)
            final_star_vels = np.array([vels_x, vels_y, vels_z]).T
            final_star_pIDs = star_pIDs[final_inds]
            final_sfts = sfts[final_inds]


            return final_gas_coords, final_gas_masses, final_gas_smoothing_lengths, \
                final_gas_densities, final_gas_temps, final_gas_vels, final_gas_cs, \
                final_gas_int_energy, final_temps, final_gas_pIDs, final_star_coords, final_star_masses, \
                    final_star_pIDs, final_star_vels, final_sfts, cloud_box

        else:
            return final_gas_coords, final_gas_masses, final_gas_smoothing_lengths, \
                    final_gas_densities, final_gas_temps, final_gas_vels, final_gas_cs, \
                        final_gas_int_energy, final_gas_pIDs, cloud_box

    return None





########################################################################################
########################### Getting cloud population quantities   ######################
########################################################################################


class CloudPopData():
    def __init__(self, cloud_total_masses, cloud_reffs, cloud_centres, cloud_hmrads, \
                 cloud_num_parts, cloud_virs, z_dists, r_gal_dists, cloud_nums):
        self.cloud_total_masses = cloud_total_masses
        self.cloud_reffs = cloud_reffs
        self.cloud_centres = cloud_centres
        self.cloud_hmrads = cloud_hmrads
        self.cloud_num_parts = cloud_num_parts
        self.cloud_virs = cloud_virs
        self.z_dists = z_dists
        self.r_gal_dists = r_gal_dists
        self.cloud_nums = cloud_nums

    
def get_cloud_pop_data(params, snapnum, nmin, vir, mw_cut=False):
    """ 
    Function to get the cloud population data for a snapshot. 

    Inputs:
        params: Parameters of the simulation.
        snapnum: Snapshot number.
        nmin: Minimum number of particles in a cloud.
        vir: Virial parameter.
    
    Outputs:
        cloud_pop_data: Cloud population data.
    """
    cloud_total_masses, cloud_reffs, cloud_centress, cloud_hmrads, \
        cloud_num_parts, cloud_virs, z_dists, r_gal_dists, cloud_nums = \
    read_cloud_summary_data(params, snapnum, params.r_gal, params.h, \
                            MW_cut=mw_cut)
    cloud_pop_data = CloudPopData(cloud_total_masses, cloud_reffs, cloud_centress, \
                                  cloud_hmrads, cloud_num_parts, cloud_virs, z_dists, r_gal_dists, cloud_nums)
    return cloud_pop_data


def read_cloud_summary_data(params, snapnum, r_gal, h, MW_cut=False):
    """ 
    Function to read the cloud summary data for a snapshot.

    Inputs:
        path: Path to the data.
        snapnum: Snapshot number.
        nmin: Minimum number of particles in a cloud.
        vir: Virial parameter.
        params: Parameters of the simulation.
        r_gal: Radius of the galaxy.
        h: Height of the galaxy.
        MW_cut: If True, only clouds within the galaxy are selected.
    
    Outputs:
        cloud_total_masses: Total mass of the clouds.
        cloud_reffs: Effective radii of the clouds.
        cloud_centress: Centres of the clouds.
        cloud_hmrads: Half-mass radii of the clouds.
        cloud_num_parts: Number of particles in the clouds.
        cloud_virs: Virial parameters of the clouds.
        z_dists: Distances of the clouds from the galactic plane.
        r_gal_dists: Distances of the clouds from the galactic center.
    """
    #r_gal = 25
    #h = 3
    path = params.path+'CloudPhinderData/n{nmin}_alpha{vir}/'.format(nmin=params.nmin, vir=params.vir)
    filename = 'bound_{snap_num}_n{nmin}_alpha{vir}.dat'.format(snap_num = snapnum, nmin = params.nmin, vir=params.vir)
    datContent = [i.strip().split() for i in open(path+filename).readlines()]
    
    cloud_total_mass_list = []
    cloud_centre_list = []
    cloud_reff_list = []
    cloud_hmrad_list = []
    cloud_num_part_list = []
    cloud_vir_list = []
    z_dists = []
    r_gal_dists = []
    cloud_nums = []

    for i in range (0, len(datContent)):
        j = i-params.dat_file_header_size
        if (i<params.dat_file_header_size):
            continue
        cloud_total_mass = float(datContent[i][0])
        cloud_centre_x = float(datContent[i][1])
        cloud_centre_y = float(datContent[i][2])
        cloud_centre_z = float(datContent[i][3])
        cloud_reff = float(datContent[i][7])
        cloud_hmrad = float(datContent[i][8])
        cloud_num_part = float(datContent[i][9])
        cloud_vir = float(datContent[i][10])
        
        if cloud_num_part<32:
            continue
        
        
        gal_centre = get_galaxy_centre(params, snapnum)
        dist = np.sqrt((cloud_centre_x-gal_centre[0])**2+(cloud_centre_y-gal_centre[1])**2+\
                      (cloud_centre_z-gal_centre[2])**2)
        
        cloud_centre = np.array([cloud_centre_x, cloud_centre_y, cloud_centre_z])
        
        ## z-distance from galactic center
        proj = get_galaxy_proj_matrix(params, snapnum)
        cloud_centre_proj = np.matmul(proj, cloud_centre)
        gal_centre_proj = np.matmul(proj, gal_centre)

        z_dist = cloud_centre_proj[2] - gal_centre_proj[2]
        
        if MW_cut:
            if dist>=r_gal or np.abs(z_dist)>=h:
                continue
        #if dist>=r_gal or np.abs(z_dist)>=h:
        #    continue
        #print (cloud_centre, cloud_centre_proj, gal_centre, gal_centre_proj, z_dist)
        #cloud_z_dists.append(z_dist)
        #cloud_inds.append(i)
        
        
        #if dist>=25:
        #    continue

        cloud_total_mass_list.append(cloud_total_mass)
        cloud_centre_list.append(np.array([cloud_centre_proj[0], cloud_centre_proj[1], cloud_centre_proj[2]]))
        cloud_reff_list.append(cloud_reff)
        cloud_hmrad_list.append(cloud_hmrad)
        cloud_num_part_list.append(cloud_num_part)
        cloud_vir_list.append(cloud_vir)
        z_dists.append(z_dist)
        r_gal_dists.append(dist)
        cloud_nums.append(j)
    
    cloud_total_masses = np.array(cloud_total_mass_list)
    cloud_reffs = np.array(cloud_reff_list)
    cloud_centress = np.array(cloud_centre_list)
    cloud_hmrads = np.array(cloud_hmrad_list)
    cloud_num_parts = np.array(cloud_num_part_list)
    cloud_virs = np.array(cloud_vir_list)
    z_dists = np.array(z_dists)
    r_gal_dists = np.array(r_gal_dists)
    cloud_nums = np.array(cloud_nums)
    
    #print (cloud_num_parts)
    
    return cloud_total_masses, cloud_reffs, cloud_centress, \
        cloud_hmrads, cloud_num_parts, cloud_virs, z_dists, \
            r_gal_dists, cloud_nums



