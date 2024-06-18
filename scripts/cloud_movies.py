
import sys
sys.path.insert(0, '../src/')
#from IC_funcs import *
from epsff_calc import *
from viz_utils import *
from cloud_selector_funcs import *
from fire_utils import *
#from compute_cloud_list import *

import yt
from yt.units import parsec, Msun
from yt.utilities.cosmology import Cosmology


from matplotlib.colors import LogNorm
from matplotlib import colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def get_stars_from_snap(params, snap_num):
    coords = Load_FIRE_Data('Coordinates', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
    masses = Load_FIRE_Data('Masses', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
    sfts = Load_FIRE_Data('StellarFormationTime', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
    
    f = h5py.File(params.path+'snapdir_{num}/snapshot_{num}.0.hdf5'.format(num=snap_num), 'r')
    hubble_constant = f['Header'].attrs['HubbleParam']
    omega_matter = f['Header'].attrs['Omega0']
    omega_lambda = f['Header'].attrs['OmegaLambda']
    current_time = f['Header'].attrs['Time']
    f.close()
    co = Cosmology(hubble_constant=hubble_constant, \
                    omega_matter=omega_matter, omega_lambda=omega_lambda)
    

    scales = sfts
    ages = np.array(co.t_from_a(current_time).in_units('Myr') - co.t_from_a(scales).in_units('Myr'))

    return coords, masses, ages


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
        
        _, _, cloud_reff, _, _ = get_cloud_info(params.path, snap_num, \
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
    dens = load_from_snapshot.load_from_snapshot('Density', 0, snapdir, snap_num, units_to_physical=False)
    gizmoSFR = load_from_snapshot.load_from_snapshot('StarFormationRate', 0, snapdir, snap_num, units_to_physical=False)
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
    dens = dens[indices]
    gizmoSFR = gizmoSFR[indices]
    # We have to clean the data a bit, there are still particles far away from the cloud that have the same pID, gen_num, child_num.

    #Find the median x, y and z coordinates from coords and exclude particles that are more than 4 kpc away from the median.
    max_reff, max_reff_cloud_num, max_reff_snap_num, cloud_reff_factor = get_maximum_cloud_reff(chain, params)

    median_coords = np.median(coords, axis=0)
    distances = np.linalg.norm(coords - median_coords, axis=1)
    new_indices = np.where(distances < 2*max_reff)[0]
    coords = coords[new_indices]
    vels = vels[new_indices]
    masses = masses[new_indices]
    dens = dens[new_indices]
    gizmoSFR = gizmoSFR[new_indices]

    # Now we can calculate the center of mass
    COM_coords = np.average(coords, axis=0, weights=masses)
    COM_vels = np.average(vels, axis=0, weights=masses)

    return COM_coords, COM_vels, coords, dens, gizmoSFR, masses



def get_cloud_props(cloud_num, snap_num, params):
    """
    Get properties of the cloud to put in the plot

    Quantities:
    - Convex hull boundaries -- calculate here
    - Surface density -- calculate here
    - Mach number -- calculate here
    - Virial parameter -- retrieve here
    - Mass -- retrieve here
    - Radius -- retrieve here
    - tff -- calculate here
    - SFR -- from the saved data
    """
    dens, vels, coords, masses, hsml, c_s, temps = get_cloud_quants_hdf5(cloud_num, snap_num, params)
    ## Find the convex hull of the cloud
    hull = ConvexHull(coords[:,:2])
    bound_x = np.append(coords[hull.vertices,0], coords[hull.vertices[0], 0])
    bound_y = np.append(coords[hull.vertices,1], coords[hull.vertices[0], 1])
    
    ## Find the free-fall time
    multi_tff = 1/(np.sum(masses*np.sqrt(32*G*1e10*dens/(3*np.pi)))/np.sum(masses))  ## in Myr

    ## Find the surface density
    sigma_gas = np.sum(masses)*1e10/(hull.volume*1e6)  ## in Msun/pc^2
    
    ## Find the Mach number
    vel_com_x = np.sum(masses*vels[:,0])/np.sum(masses)
    vel_com_y = np.sum(masses*vels[:,1])/np.sum(masses)
    vel_com_z = np.sum(masses*vels[:,2])/np.sum(masses)
    vel_x = vels[:,0]
    vel_y = vels[:,1]
    vel_z = vels[:,2]
    vsq = (vel_x-vel_com_x)**2+(vel_y-vel_com_y)**2+(vel_z-vel_com_z)**2
    vel_disp = np.sqrt(np.sum(vsq)/len(vsq)/3)  #1-D velocity dispersion
    cs_mw = np.sqrt(np.sum(masses*c_s**2)/np.sum(masses)) #Mass weighted sound speed
    mach_number = vel_disp/cs_mw

    ## Find the virial parameter
    _, cloud_centre, cloud_reff, cloud_vir, _ = get_cloud_info(params.path, snap_num, \
                                                            params.nmin, params.vir, cloud_num, params)

    ## Find the SFR
    sfr = get_sfr(params, snap_num, cloud_centre, cloud_reff)
    

    return bound_x, bound_y, sigma_gas, mach_number, np.sum(masses)*1e10, cloud_reff, cloud_vir, multi_tff, sfr



def get_sfr(params, snap_num, cloud_centre, cloud_reff):
    age_file = params.path+params.star_data_sub_dir+'Ages_cut{cut}_{snap}.npz'.format(cut=params.age_cut,\
                                                                                        snap=snap_num)
    ages_data = np.load(age_file)
    ages = ages_data['arr_0']
    
    mass_file = params.path+params.star_data_sub_dir+'Masses_cut{cut}_{snap}.npz'.format(cut=params.age_cut,\
                                                                                        snap=snap_num)
    masses_data = np.load(mass_file)
    masses = masses_data['arr_0']
    
    coords_file = params.path+params.star_data_sub_dir+'Coords_cut{cut}_{snap}.npz'.format(cut=params.age_cut,\
                                                                                        snap=snap_num)
    coords_data = np.load(coords_file)
    young_star_x = coords_data['arr_0']
    young_star_y = coords_data['arr_1']
    young_star_z = coords_data['arr_2']
    #young_star_coords = []
    #for i in range(len(young_star_x)):
    #    young_star_coords.append(np.array([young_star_x[i], young_star_y[i], young_star_z[i]]))

    #young_star_coords=np.array(young_star_coords)
    
    n_stars = 0
    young_star_mass = 0
    for j in range (0, len(young_star_x)):
        dist = np.sqrt((young_star_x[j]-cloud_centre[0])**2 +\
                       (young_star_y[j]-cloud_centre[1])**2 +\
                       (young_star_z[j]-cloud_centre[2])**2)

        if dist<=cloud_reff:
            n_stars = n_stars+1
            young_star_mass = masses[j] + young_star_mass
    
    cloud_num_stars = n_stars

    return young_star_mass/params.age_cut









def make_movies_of_clouds(params, selected_cloud_nums, selected_cloud_list, selected_snap):
    
    #chain = CloudChain(selected_cloud_nums, selected_snap, params)    
    #for snap_num in range(params.start_snap, params.start_snap+1):
    #for snap_num in range(selected_snap, selected_snap+1):
    for snap_num in range(params.start_snap, params.last_snap+1):
        print ("Snap Num: ", snap_num)
        # Load the snapshot data
        snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
        coords = load_from_snapshot.load_from_snapshot('Coordinates', 0, snapdir, snap_num, units_to_physical=True)
        masses = load_from_snapshot.load_from_snapshot('Masses', 0, snapdir, snap_num, units_to_physical=True)
        vels = load_from_snapshot.load_from_snapshot('Velocities', 0, snapdir, snap_num, units_to_physical=True)
        dens = load_from_snapshot.load_from_snapshot('Density', 0, snapdir, snap_num, units_to_physical=True)
        hsmls = load_from_snapshot.load_from_snapshot('SmoothingLength', 0, snapdir, snap_num, units_to_physical=True)
        gizmoSFR = load_from_snapshot.load_from_snapshot('StarFormationRate', 0, snapdir, snap_num, units_to_physical=True)
        pIDs = load_from_snapshot.load_from_snapshot('ParticleIDs', 0, snapdir, snap_num, units_to_physical=True)
        pID_gen_nums = load_from_snapshot.load_from_snapshot('ParticleIDGenerationNumber', 0, snapdir, snap_num, units_to_physical=True)
        pID_child_nums = load_from_snapshot.load_from_snapshot('ParticleChildIDsNumber', 0, snapdir, snap_num, units_to_physical=True)
        pID_array = np.column_stack((pIDs, pID_gen_nums, pID_child_nums))
        print ("Loaded gas data")

        star_coords, star_masses, star_ages = get_stars_from_snap(params, snap_num)
        print ("Loaded star data")

        cloud_count = 0
        for cloud_num in selected_cloud_nums:
            #cloud_num = int(cloud.split('Snap')[0].split('Cloud')[1])
            # We will 
            chain = CloudChain(cloud_num, selected_snap, params)    
            tracked_cloud_pID_array = get_tracked_cloud_pIDs(chain, params)            
            # Get the indices of the particles we want -- basically the ones which are in the tracked_cloud_pID_array
            check = np.isin(pID_array, tracked_cloud_pID_array)
            indices = np.where(check.all(axis=1))[0]
            #indices = np.where(np.in1d(pID_array, tracked_cloud_pID_array).reshape(pID_array.shape))[0]
            cloud_coords = coords[indices]
            cloud_vels = vels[indices]
            cloud_masses = masses[indices]
            cloud_hsmls = hsmls[indices]
            cloud_dens = dens[indices]
            cloud_gizmoSFR = gizmoSFR[indices]

            # We have to clean the data a bit, there are still particles far away from the cloud that have the same pID, gen_num, child_num.

            #Find the median x, y and z coordinates from coords and exclude particles that are more than 4 kpc away from the median.
            max_reff, max_reff_cloud_num, max_reff_snap_num, cloud_reff_factor = get_maximum_cloud_reff(chain, params)

            median_coords = np.median(cloud_coords, axis=0)
            distances = np.linalg.norm(cloud_coords - median_coords, axis=1)
            new_indices = np.where(distances < 2*max_reff)[0]
            cloud_coords = cloud_coords[new_indices]
            cloud_vels = cloud_vels[new_indices]
            cloud_masses = cloud_masses[new_indices]
            cloud_hsmls = cloud_hsmls[new_indices]
            cloud_dens = cloud_dens[new_indices]
            cloud_gizmoSFR = cloud_gizmoSFR[new_indices]

            # Get the bounding box for these particles 
            x_min, x_max, y_min, y_max, z_min, z_max = cloud_coords[:,0].min(), cloud_coords[:,0].max(), \
                                                        cloud_coords[:,1].min(), cloud_coords[:,1].max(), \
                                                        cloud_coords[:,2].min(), cloud_coords[:,2].max()
            
            box_size = max([x_max-x_min, y_max-y_min, z_max-z_min])
            #box_size = box_size
            x_mean = (x_min+x_max)/2
            y_mean = (y_min+y_max)/2
            z_mean = (z_min+z_max)/2

            # Now we will get all the gas particles and star particles within this box from the snapshot data.
            inds_x = np.where((coords[:,0]>x_mean-box_size/2)&(coords[:,0]<x_mean+box_size/2))[0]
            inds_y = np.where((coords[:,1]>y_mean-box_size/2)&(coords[:,1]<y_mean+box_size/2))[0]
            inds_z = np.where((coords[:,2]>z_mean-box_size/2)&(coords[:,2]<z_mean+box_size/2))[0]
            final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)
        
            coords_x = np.take(coords[:,0], final_inds)
            coords_y = np.take(coords[:,1], final_inds)
            coords_z = np.take(coords[:,2], final_inds)
            final_gas_coords = np.array([coords_x, coords_y, coords_z]).T
            final_gas_masses = np.take(masses, final_inds)
            final_gas_hsmls = np.take(hsmls, final_inds)

            # Get the star particles
            inds_x = np.where((star_coords[:,0]>x_mean-box_size/2)&(star_coords[:,0]<x_mean+box_size/2))[0]
            inds_y = np.where((star_coords[:,1]>y_mean-box_size/2)&(star_coords[:,1]<y_mean+box_size/2))[0]
            inds_z = np.where((star_coords[:,2]>z_mean-box_size/2)&(star_coords[:,2]<z_mean+box_size/2))[0]
            final_inds = np.intersect1d(np.intersect1d(inds_x, inds_y), inds_z)

            coords_x = np.take(star_coords[:,0], final_inds)
            coords_y = np.take(star_coords[:,1], final_inds)
            coords_z = np.take(star_coords[:,2], final_inds)
            final_star_coords = np.array([coords_x, coords_y, coords_z]).T
            final_star_masses = np.take(star_masses, final_inds)
            final_star_ages = np.take(star_ages, final_inds)

            make_cloud_plot(final_gas_coords, final_gas_masses, final_gas_hsmls, final_star_coords, final_star_masses, final_star_ages, \
                            params, box_size, selected_cloud_list, cloud_count, snap_num, x_mean, y_mean, z_mean, max_reff, chain)
            cloud_count += 1

    return

def make_cloud_plot(final_gas_coords, final_gas_masses, final_gas_hsmls, final_star_coords, final_star_masses, final_star_ages, \
                            params, box_size, selected_cloud_list, cloud_count, snap_num, x_mean, y_mean, z_mean, max_reff, chain):
    cb_vmin = 1e-1
    cb_vmax = 5e3

    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)

    res = 1024
    center = np.array([x_mean, y_mean, z_mean])
    image_box_size = box_size
    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    M = Meshoid(final_gas_coords, final_gas_masses, final_gas_hsmls)
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)*1e4    
    print ('Computed surface density ...')
    
    # Plot the cloud
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=cb_vmin,vmax=cb_vmax), cmap = 'inferno')

    ax.set_aspect('equal')
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1], max_pos[1]])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Plot the gas particles 
    ax.scatter(final_gas_coords[:,0], final_gas_coords[:,1], s=0.1, c='w', alpha=0.5)

    # Plot the cloud particles
    if snap_num in chain.snap_nums:
        cnum = chain.cloud_nums[chain.snap_nums.index(snap_num)]
        _, _, coords, _, _, _, _ = get_cloud_quants_hdf5(cnum, snap_num, params)
        # Plot the coords
        ax.scatter(coords[:,0], coords[:,1], s=0.2, c='k', alpha=0.5)

        ## Find the convex hull of the cloud
        hull = ConvexHull(coords[:,:2])
        bound_x = np.append(coords[hull.vertices,0], coords[hull.vertices[0], 0])
        bound_y = np.append(coords[hull.vertices,1], coords[hull.vertices[0], 1])
        # Plot the cloud convex hull
        plt.plot(bound_x, bound_y, color='black', lw=1, linestyle='--')

    # Plot the stars
    age_cut = 5
    young_star_coords_x = final_star_coords[:,0][final_star_ages<age_cut]
    young_star_coords_y = final_star_coords[:,1][final_star_ages<age_cut]
    young_star_coords = np.array([young_star_coords_x, young_star_coords_y]).T

    #print (young_star_coords, final_gas_coords)
    # Let's color the stars by their age using the blues colormap
    ax.scatter(young_star_coords[:,0], young_star_coords[:,1], s=50, c=final_star_ages[final_star_ages<age_cut], marker='*', cmap='bwr', alpha=1)


    # Plot the cloud convex hull
    #plt.plot(bound_y-cloud_centre[1], bound_x-cloud_centre[0], color='black', lw=1, linestyle='--')


    ## Add cloud data now
    #alpha = 0.5
    #cref = cloud_reff*cloud_reff_factor
    #ax.text(-0.95*cref,-0.2*cref, r"$\rm R_{gal} = %.2f\,kpc$" % (r_gal), fontsize=16, color='w', alpha=alpha)
    #ax.text(-0.95*cref,-0.35*cref, r"$\rm SFR = %.4f\,M_\odot\,yr^{-1}$" % (sfr*1e4), fontsize=16, color='w', alpha=alpha)
    #ax.text(-0.95*cref,-0.5*cref, r"$\rm M_6 = %.2f\,M_\odot$" % (cloud_total_mass/1e6), fontsize=16, color='w', alpha=alpha)
    #ax.text(-0.95*cref,-0.65*cref, r"$\rm t_{ff, mult} = %.2f\,Myr$" % (multi_tff), fontsize=16, color='w', alpha=alpha)
    #ax.text(-0.95*cref,-0.8*cref, r"$\rm R = %.2f\,pc$" % (cloud_reff*1e3), fontsize=16, color='w', alpha=alpha)
    #ax.text(-0.95*cref,-0.95*cref, r"$\rm \Sigma_{gas} = %.2f\,M_\odot\,pc^{-2}$" % (sigma_gas), fontsize=16, color='w', alpha=alpha)
    #ax.text(0.2*cref, 0.8*cref, r"$\rm t=%.2f\,Myr$" % (snap_num-params.start_snap), fontsize=16, color='w', alpha=1)
    #ax.text(0.2*cref,0.65*cref, r"$\rm \mathcal{M} = %.2f$" % (mach_number), fontsize=16, color='w', alpha=alpha)
    #ax.text(0.2*cref,0.5*cref, r"$\rm \alpha_{vir} = %.2f$" % (cloud_vir), fontsize=16, color='w', alpha=alpha)



    #bound_x, bound_y, sigma_gas, mach_number, cloud_total_mass, cloud_reff, cloud_vir, multi_tff, sfr = get_cloud_props(cloud_num, snap_num, params)  
    #ax.text()
    #, transform=ax.transAxes)

    
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                                max_reff, '%.2f kpc' %(max_reff), 'upper left', 
                                pad=1,
                                color='white',
                                frameon=False,
                                size_vertical=0.005,
                                fontproperties=fontprops)
    ax.add_artist(scalebar)

    ax.set_xticks([])
    ax.set_yticks([])

    
    #cb_ax = fig.add_axes([0.9, 0.125, 0.01, 0.755])
    #cbar = fig.colorbar(p, cax=cb_ax, orientation="vertical",\
                        #ticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])#, label='Counts', labelsize=22)

    #cbar.set_label(label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$", size=22)
    #cbar = fig.colorbar(h[3], ax=ax[0,1])#, labelsize=20)
    #cbar.ax.tick_params(labelsize=20)#, fontsize=22)
    #Check if directory exists, if not create it.
    print ("Looking for directory: ", "../images/SelectedCloudChain0-3/{cloud}".format(cloud=selected_cloud_list[cloud_count]))
    if not os.path.exists("../images/SelectedCloudChain0-3/{cloud}".format(cloud=selected_cloud_list[cloud_count])):
        print ("Creating directory: ", "../images/SelectedCloudChain0-3/{cloud}".format(cloud=selected_cloud_list[cloud_count]))
        os.makedirs("../images/SelectedCloudChain0-3/{cloud}".format(cloud=selected_cloud_list[cloud_count]))
    
    plt.tight_layout()
    plt.savefig(f"../images/SelectedCloudChain0-3/{selected_cloud_list[cloud_count]}/Snap_{snap_num}.png")
    #plt.show()
    plt.close()

    return 



path = "../../../FIRE-2/m12i_final/"

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
frac_thresh='thresh0.3'

r_gal = 25
h = 0.4 #0.4

save_path = './data/'
#snapnum = 650


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





#selected_cloud_nums = [cloud_list[i]['cloud_num'] for i in unique_inds]
selected_cloud_nums = np.array([16, 65, 77, 164, \
                                114, 25, 97, 146, \
                                106, 40, 10, 83, 109])
selected_cloud_list = ['Cloud0016Snap625', 'Cloud0065Snap625', \
                        'Cloud0077Snap625', 'Cloud0164Snap625', \
                        'Cloud0114Snap625', 'Cloud0025Snap625', \
                        'Cloud0097Snap625', 'Cloud0146Snap625', \
                        'Cloud0106Snap625', 'Cloud0040Snap625', \
                        'Cloud0010Snap625', 'Cloud0083Snap625', \
                        'Cloud0109Snap625']




plots_path = '../images/SelectedCloudChain0-3/'
if not os.path.exists(plots_path):
    os.makedirs(plots_path)



selected_snap = 625


print ("Making movies of clouds ...")
make_movies_of_clouds(params, selected_cloud_nums, selected_cloud_list, selected_snap)
print ("Done making movies of clouds ...")