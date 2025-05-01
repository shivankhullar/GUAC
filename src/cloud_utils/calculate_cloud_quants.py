"""
This file contains functions to get individual cloud quantities like the mach number, sfr, etc. for a 
given snapshot.

Author: Shivan Khullar
Date: June 2024
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
import scipy

#### ---------------------------------------------------------------------- ####
#### ---------------------------------------------------------------------- ####
#### Functions to calculate things related to the b parameter of the clouds ####
#### ---------------------------------------------------------------------- ####
#### ---------------------------------------------------------------------- ####


def calculate_cloud_mach(vels, masses, c_s):
    """ 
    Function to calculate the Mach number of a cloud.
    """
    # Mass-weighted velocity dispersion
    vel_disp_1D = get_velocity_dispersion(vels, masses)
    # Mass-weighted sound speed
    mean_cs = np.sqrt(np.sum(masses*c_s**2)/np.sum(masses))
    mach = vel_disp_1D/mean_cs
    return mach

def calculate_mach_turb(masses, coords, vels, cs):
    com_coords = np.array([np.sum(masses*coords[:,0])/np.sum(masses), \
                        np.sum(masses*coords[:,1])/np.sum(masses), \
                        np.sum(masses*coords[:,2])/np.sum(masses)])

    coords = coords - com_coords

    com_vel = np.array([np.sum(masses*vels[:,0])/np.sum(masses), \
                        np.sum(masses*vels[:,1])/np.sum(masses), \
                            np.sum(masses*vels[:,2])/np.sum(masses)])

    #print ('COM vel:', com_vel)
    vels = vels - com_vel

    total_KE = 0.5*np.sum(masses*(vels[:,0]**2+vels[:,1]**2+vels[:,2]**2))

    dists = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2)
    #Rotation vector for each particle
    omega = np.zeros((len(dists), 3))
    for i in range(0, len(dists)):
        omega[i] = np.cross(coords[i], vels[i]) #/dists[i]**2
    #Angular velocity vector for the whole cloud
    omega_cloud_x = np.sum(masses*omega[:,0])/np.sum(masses*dists[i]**2)
    omega_cloud_y = np.sum(masses*omega[:,1])/np.sum(masses*dists[i]**2)
    omega_cloud_z = np.sum(masses*omega[:,2])/np.sum(masses*dists[i]**2)
    omega_cloud = np.array([omega_cloud_x, omega_cloud_y, omega_cloud_z])

    v_rot = np.zeros((len(dists), 3))
    for i in range(0, len(dists)):
        v_rot[i] = np.cross(omega_cloud, coords[i])


    solid_body_rot_KE = 0.5*np.sum(masses*(v_rot[:,0]**2+v_rot[:,1]**2+v_rot[:,2]**2))

    turb_KE = total_KE - solid_body_rot_KE
    #There's one cloud for which turb KE is negative apparently. Fuck that cloud. 
    if turb_KE<0:
        turb_KE = 0

    turb_vel_disp_1D = np.sqrt(2*turb_KE/np.sum(masses)/3)   #Factor of 3 for 1D velocity dispersion
    cs_mean = np.sqrt(np.sum(masses*cs**2)/np.sum(masses))
    mach_turb = turb_vel_disp_1D/cs_mean
    if mach_turb == 0:
        mach_turb = 1e-5

    return mach_turb, turb_vel_disp_1D, solid_body_rot_KE, total_KE, turb_KE



def get_velocity_dispersion(vels, masses):
    """
    Function to get the (mass-weighted) velocity dispersion of a cloud.
    """
    #dens, vels, coords, masses, hsml, cs = get_cloud_quants_hdf5(cloud_num, snapnum, params)
    #n, bins, _ = np.hist(np.log10(dens), bins=100, density=True, alpha=0.75)
    vel_com_x = np.sum(masses*vels[:,0])/np.sum(masses)
    vel_com_y = np.sum(masses*vels[:,1])/np.sum(masses)
    vel_com_z = np.sum(masses*vels[:,2])/np.sum(masses)
    vel_x = vels[:,0]
    vel_y = vels[:,1]
    vel_z = vels[:,2]
    vsq = (vel_x-vel_com_x)**2+(vel_y-vel_com_y)**2+(vel_z-vel_com_z)**2
    #vel_disp = np.sqrt(np.sum(vsq)/len(vsq))
    vel_disp = np.sqrt(np.sum(vsq)/len(vsq)/3)  #1-D velocity dispersion
    
    return vel_disp

def get_convex_hull_volume(coords):
    """
    Function to get the volume of the convex hull of a set of points.
    """
    hull = scipy.spatial.ConvexHull(coords)
    return hull.volume


def find_percentile_var(data):
    """ 
    Function to find the variance of the density PDF. 
    Variance here is the 16th-84th percentile range.
    """
    perc_16 = np.percentile(data, 16)
    perc_84 = np.percentile(data, 84)
    return (perc_84-perc_16)/2


def calculate_density_pdf_variance(dens, weights=None, s_bins=None):
    """
    Function to calculate the variance in the density PDF of a cloud. \
        Mean density is always mass-weighted unless weights are provided.
    """
    if weights is not None:
        mean_dens = np.sum(dens*weights)/np.sum(weights)
        s = np.log(dens/mean_dens)
    else:
        s = np.log(dens/np.mean(dens))

    #if weights is None:
    #    sigma = find_percentile_var(s)
    #else:
    #    s_weighted = np.log(dens*weights/np.sum(weights)/mean_dens)
    #    sigma = find_percentile_var(s_weighted)
    
    if s_bins is None:
        s_bins = np.linspace(-4, 6, 101)
    
    n, binned = np.histogram(s, bins = s_bins, weights=weights, density = True)
    cdf = np.cumsum(n)/np.sum(n)
    bin_centers = (binned[1:]+binned[:-1])/2
    percentiles = np.interp([0.16, 0.84], cdf, bin_centers)
    sigma = (percentiles[1]-percentiles[0])/2

    ## This was just to calculate variance. For storing the pdf we'll use a broader range. 
    ## Things anyway reach 1 in the cdf by s=2, so we're just increasing resolution for finding
    ## percentiles more accurately.
    ## For plotting the PDF, we'll use a broader range.
    if s_bins is None:
        s_bins = np.linspace(-4, 8, 101)
    n, binned = np.histogram(s, bins = s_bins, weights=weights, density = True)
    bin_centers = (binned[1:]+binned[:-1])/2
    return sigma, n, bin_centers

def get_epsff_cloud(dens, gas_masses, sfr):
    """ 
    Function to calculate the star formation efficiency per free-fall time of a cloud.

    Inputs:
        dens: Density of the cloud.
        gas_masses: Masses of the gas particles in the cloud.
        star_masses: Masses of the star particles in the cloud.
    
    Outputs:
        eps_ff: Star formation efficiency per free-fall time.
    """
    kpc = 3.0857e21
    Msun = 1.989e33
    yr = 365.25*24*3600
    Myr = 1e6*(yr)
    G = 6.67430e-8  ## in cgs -- cm^3 g^-1 s^-2
    G = G*Msun*Myr**2/kpc**3
    multi_tff = 1/(np.sum(gas_masses*np.sqrt(32*G*1e10*dens/(3*np.pi)))/np.sum(gas_masses)) # in Myr
    eps_ff = sfr/(np.sum(gas_masses)*1e10/multi_tff)     # in Msun/Myr over Msun/Myr, so unitless
    return eps_ff



def get_cloud_pop_quant_info(cloud_inds, snapnum, params, cloud_pop_data, get_from_cloud_box=False, snap_data=None):
    """
    Function to calculate the cloud properties.

    Inputs:
        cloud_inds: Indices of the clouds.
        snapnum: Snapshot number.
        params: Parameters of the simulation.
        cloud_pop_data: Cloud population data.
        get_from_cloud_box: If True, we will calculate quantities using all the particles in the cloud box.
        snap_data: Snapshot data.

    Outputs:
        cloud_list: List of dictionaries containing the cloud properties.
    """
    cloud_list = []
    for i in range(len(cloud_inds)):
        cloud_quants = dict.fromkeys(['cloud_num','masses', 'dens', 'vels', 's_dens',  \
                                'coords', 'hsml', 'c_s', 'box_size', 'total_mass', \
                                'gal_dist', 'z_dist', 'r_eff', 'hmrad', 'vel_disp', 'vir', \
                                'mach', 'b_param_vol', 'b_param_turb', 'sigma_mass', 'sigma_vol', \
                                'sigma', 'convex_hull_vol', 'c_s', 'c_s_mean', 'temps', \
                                'n_vol', 'bin_centers', 'vel_disp_turb', 'mach_turb', \
                                'star_masses', 'star_coords', 'sfr', 'eps_ff', 'total_KE',\
                                    'rot_KE', 'turb_KE'])
    
        #cloud_num = cloud_inds[i]
        cloud_ind = cloud_inds[i]
        cloud_num = cloud_pop_data.cloud_nums[cloud_ind]
        if get_from_cloud_box == True:
            # We will calculate quantities using all the particles in the cloud box
            coords, masses, hsml, dens, _, vels, cs, int_energy, temps, gas_pIDs, star_coords, star_masses, \
                star_pIDs, star_vels, sfts, cloud_box = get_cloud_quants(cloud_num, snapnum, \
                                            params, cloud_reff_factor=2, cloud_box=None, \
                                            snap_data=snap_data, star_data=True)
        else:
            # We won't calculate quantities using all the particles in the cloud box
            if snap_data is not None:
                dens, vels, coords, masses, hsml, cs, temps, star_coords, star_masses = get_cloud_quants(cloud_num, snapnum, \
                                                    params, cloud_reff_factor=2, cloud_box=False,\
                                                        snap_data=snap_data, star_data=True, pID_mode=True)
            else:
                dens, vels, coords, masses, hsml, cs, temps, star_coords, star_masses = get_cloud_quants(cloud_num, snapnum, \
                                                    params, cloud_reff_factor=2, cloud_box=False,\
                                                        snap_data=None, star_data=True)


        volumes = masses/dens
        #assert np.sum(masses)==cloud_pop_data.cloud_total_masses[cloud_num]
        #print (np.sum(masses), cloud_pop_data.cloud_total_masses[cloud_num])
        vel_disp = get_velocity_dispersion(vels, masses)
        mach = calculate_cloud_mach(vels, masses, cs)
        #vels_turb_x = vels[:,0] - vels[:,0].mean()
        #vels_turb_y = vels[:,1] - vels[:,1].mean()
        #vels_turb_z = vels[:,2] - vels[:,2].mean()
        #vels_turb = np.array([vels_turb_x, vels_turb_y, vels_turb_z]).T
        #vel_disp_turb = get_velocity_dispersion(vels_turb, masses)

        mach_turb, turb_vel_disp, rot_KE, total_KE, turb_KE = calculate_mach_turb(masses, coords, vels, cs)

        sigma_mass, n_mass, bin_centers = calculate_density_pdf_variance(dens, weights=masses)
        sigma_vol, n_vol, bin_centers = calculate_density_pdf_variance(dens, weights=volumes)
        sigma, n, bin_centers = calculate_density_pdf_variance(dens, weights=None)

        cloud_quants['cloud_num'] = cloud_num
        if get_from_cloud_box == True:
            cloud_quants['box_size'] = cloud_box['x_max'] - cloud_box['x_min']
        cloud_quants['masses'] = masses
        cloud_quants['dens'] = dens
        mean_dens = np.sum(dens*masses)/np.sum(masses)
        s = np.log(dens/mean_dens)
        cloud_quants['s_dens'] = s
        cloud_quants['coords'] = coords
        cloud_quants['hsml'] = hsml
        cloud_quants['c_s'] = cs
        cloud_quants['vels'] = vels
        cloud_quants['temps'] = temps
        cloud_quants['total_mass'] = cloud_pop_data.cloud_total_masses[cloud_ind]
        cloud_quants['gal_dist'] = cloud_pop_data.r_gal_dists[cloud_ind]
        cloud_quants['z_dist'] = cloud_pop_data.z_dists[cloud_ind]
        cloud_quants['r_eff'] = cloud_pop_data.cloud_reffs[cloud_ind]
        cloud_quants['hmrad'] = cloud_pop_data.cloud_hmrads[cloud_ind]
        #cloud_quants['total_mass'] = cloud_pop_data.cloud_total_masses[cloud_num]
        #cloud_quants['gal_dist'] = cloud_pop_data.r_gal_dists[cloud_num]
        #cloud_quants['z_dist'] = cloud_pop_data.z_dists[cloud_num]
        #cloud_quants['r_eff'] = cloud_pop_data.cloud_reffs[cloud_num]
        #cloud_quants['hmrad'] = cloud_pop_data.cloud_hmrads[cloud_num]
        cloud_quants['vel_disp'] = vel_disp
        cloud_quants['vel_disp_turb'] = turb_vel_disp
        cloud_quants['vir'] = cloud_pop_data.cloud_virs[cloud_ind]
        #cloud_quants['vir'] = cloud_pop_data.cloud_virs[cloud_num]
        cloud_quants['mach'] = mach
        cloud_quants['mach_turb'] = mach_turb
        cloud_quants['sigma_mass'] = sigma_mass
        cloud_quants['sigma_vol'] = sigma_vol
        cloud_quants['sigma'] = sigma
        cloud_quants['convex_hull_vol'] = get_convex_hull_volume(coords)
        cloud_quants['c_s_mean'] = np.sqrt(np.sum(masses*cs**2)/np.sum(masses))
        cloud_quants['b_param_vol'] = np.sqrt((np.exp(sigma_vol**2)-1)/mach**2)
        cloud_quants['b_param_turb'] = np.sqrt((np.exp(sigma_vol**2)-1)/mach_turb**2)
        cloud_quants['n_vol'] = n_vol
        cloud_quants['bin_centers'] = bin_centers
        cloud_quants['star_masses'] = star_masses
        cloud_quants['star_coords'] = star_coords
        cloud_quants['sfr'] = np.sum(star_masses)*1e10/params.age_cut              # in Msun/Myr
        cloud_quants['eps_ff'] = get_epsff_cloud(dens, masses, np.sum(star_masses)*1e10/params.age_cut)
        cloud_quants['total_KE'] = total_KE
        cloud_quants['rot_KE'] = rot_KE
        cloud_quants['turb_KE'] = turb_KE
        

        
        cloud_list.append(cloud_quants)

        # print percentage completion
        if i%5==0:
            print ('Percent done:', i/len(cloud_inds)*100)

    return cloud_list


def calculate_cloud_quants(cloud_inds, snapnum, params):
    """
    Function to calculate the cloud properties.
    """
    # Dictionary to store cloud scale quantities: mass, radius, density PDF, velocity dispersion, Mach number, etc.
    cloud_quants = dict.fromkeys(['cloud_nums','masses', \
                             'reffs', 'density_pdfs', 'vel_disps', 'machs', 'b_params', \
                                'sigma_mass', 'sigma_vol', 'sigma', 'convex_hull_vol', 'c_s']) 
    for key in cloud_quants:
        cloud_quants[key] = []

    for i in range(len(cloud_inds)):
        cloud_num = cloud_inds[i]
        dens, vels, coords, masses, _, cs = get_cloud_quants_hdf5(cloud_num, snapnum, params)
        volumes = masses/dens
        #assert np.sum(masses)==cloud_pop_data.cloud_total_masses[cloud_num]
        #print (np.sum(masses), cloud_pop_data.cloud_total_masses[cloud_num])
        vel_disp = get_velocity_dispersion(dens, vels, masses)
        mach = calculate_cloud_mach(dens, vels, masses, cs)
        #mean_dens = np.sum(dens*masses)/np.sum(masses)
        sigma_mass = calculate_density_pdf_variance(dens, masses, weights=masses)
        sigma_vol = calculate_density_pdf_variance(dens, masses, weights=volumes)
        sigma = calculate_density_pdf_variance(dens, masses, weights=None)


        cloud_quants['cloud_nums'].append(cloud_num)
        cloud_quants['masses'].append(np.sum(masses))
        cloud_quants['reffs'].append(cloud_pop_data.cloud_reffs[cloud_num])
        #cloud_quants['density_pdfs'].append(dens)
        cloud_quants['vel_disps'].append(vel_disp)
        cloud_quants['machs'].append(mach)
        cloud_quants['sigma_mass'].append(sigma_mass)
        cloud_quants['sigma_vol'].append(sigma_vol)
        cloud_quants['sigma'].append(sigma)
        cloud_quants['convex_hull_vol'].append(get_convex_hull_volume(coords))
        cloud_quants['c_s'].append(cs.mean())

    return cloud_quants





def save_cloud_list_data(params, save_path, snapnum, prop_object, box=False):
    if box==True:
        save_filename = 'CloudBox_{sim}_{snap}_n{nmin}_a{vir}.hdf5'.format(sim=params.sim, nmin=params.nmin, vir=params.vir, snap=snapnum)
    else:
        save_filename = 'Cloud_{sim}_{snap}_n{nmin}_a{vir}.hdf5'.format(sim=params.sim, nmin=params.nmin, vir=params.vir, snap=snapnum)
    dill.dump(prop_object, file = open(save_path+save_filename+".pickle", "wb"))
    print ("Saved the cloud_list object in file: ", save_path+save_filename+".pickle")
    return


def load_cloud_list_data(params, save_path, snapnum, box=False):
    if box==True:
        save_filename = 'CloudBox_{sim}_{snap}_n{nmin}_a{vir}.hdf5'.format(sim=params.sim, nmin=params.nmin, vir=params.vir, snap=snapnum)
    else:
        save_filename = 'Cloud_{sim}_{snap}_n{nmin}_a{vir}.hdf5'.format(sim=params.sim, nmin=params.nmin, vir=params.vir, snap=snapnum)
    prop_object_reloaded = dill.load(open(save_path+save_filename+".pickle", "rb"))
    return prop_object_reloaded