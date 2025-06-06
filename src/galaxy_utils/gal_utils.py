"""
This file contains utility functions to get galaxy quantities.

Author: Shivan Khullar
Date: April 2024
"""

#import sys
#sys.path.append('/src/generic_utils/')
import generic_utils
import generic_utils.fire_utils
from generic_utils.fire_utils import *
import generic_utils.constants as const

class GalQuants():
    """
    Class to store the galaxy data and perform operations on it.

    Attributes:
        params (dict): The parameters of the simulation
        snapnum (int): The snapshot number
        inds (array_like): The indices of the particles in the galaxy
        r_gal (float): The galactocentric distance
        h (float): The height of the disk
        data (dict): The data of the galaxy
    
    Methods:
        spherical_cut(coords_data): Cut a sphere around the galaxy
        project(coords_data): Project the data into an edge-on view
        add_key(key, init_data, dim): Add a key to the galaxy data
    
            
    """
    def __init__(self, params, snapnum, r_gal, h, gal_centre=None, proj=None):
        self.params = params
        self.snapnum = snapnum
        self.inds = None
        self.r_gal = r_gal
        self.h = h
        self.data = {}
        self.gal_centre = gal_centre
        self.proj = proj
        self.verbose = params.verbose

    def spherical_cut(self, coords_data):
        if self.gal_centre is None:
            self.gal_centre = generic_utils.fire_utils.get_galaxy_centre(self.params, self.snapnum)
            #gal_centre = Get_Galaxy_Centre(self.params, self.snapnum)    
        if self.proj is None:
            self.proj = generic_utils.fire_utils.get_galaxy_proj_matrix(self.params, self.snapnum)
        #gal_centre_proj = np.matmul(proj, gal_centre)
        
        gal_dist = np.sqrt((coords_data[:,0]-self.gal_centre[0])**2+\
                            (coords_data[:,1]-self.gal_centre[1])**2+\
                       (coords_data[:,2]-self.gal_centre[2])**2)
        
        self.sphere_inds = np.where(gal_dist<=self.r_gal)[0]
        #return sphere_coords, sphere_masses

    def project(self, coords_data):
        if self.gal_centre is None:
            self.gal_centre = generic_utils.fire_utils.get_galaxy_centre(self.params, self.snapnum)
            #print ('Galactic center:', self.gal_centre)
            #gal_centre = Get_Galaxy_Centre(self.params, self.snapnum)    
        if self.proj is None:
            self.proj = generic_utils.fire_utils.get_galaxy_proj_matrix(self.params, self.snapnum)
            #print (self.proj)

            #proj = Get_Galaxy_Proj_Matrix(self.params, self.snapnum)
        self.gal_centre_proj = np.matmul(self.proj, self.gal_centre)
        #print ("Projected quantities:")
        #print (self.gal_centre_proj, self.gal_centre, self.proj)
        self.spherical_cut(coords_data)
        init_coords_x = np.take(coords_data[:,0], self.sphere_inds)
        init_coords_y = np.take(coords_data[:,1], self.sphere_inds)
        init_coords_z = np.take(coords_data[:,2], self.sphere_inds)
        gas_coords = np.array([init_coords_x, init_coords_y, init_coords_z]).T
        if self.verbose:
            print ("Starting projection...")

        proj_gas_coords = []
        for i in range(0, len(gas_coords)):
            proj_gas_coord = np.matmul(self.proj, gas_coords[i])
            proj_gas_coords.append(proj_gas_coord)
            #if i%5000==0:
                #print('Percent complete:', i/len(gas_coords)*100)
        proj_gas_coords = np.array(proj_gas_coords)
        print (proj_gas_coords, proj_gas_coords.shape)

        #if len(proj_gas_coords)==1:
            
        #    gal_dist_proj = np.sqrt((proj_gas_coords[:,0]-self.gal_centre_proj[0])**2+\
        #                            (proj_gas_coords[:,1]-self.gal_centre_proj[1])**2+\
        #                            (proj_gas_coords[:,2]-self.gal_centre_proj[2])**2)


        gal_dist_proj = np.sqrt((proj_gas_coords[:,0]-self.gal_centre_proj[0])**2+\
                            (proj_gas_coords[:,1]-self.gal_centre_proj[1])**2+\
                       (proj_gas_coords[:,2]-self.gal_centre_proj[2])**2)
        if self.verbose:
            print ("Projection done...")
        
        proj_gas_z_dist = np.abs(proj_gas_coords[:,2]-self.gal_centre_proj[2]) 
                            #np.take(np.abs(proj_gas_coords[:,2]-gal_centre_proj[2]), \
                            #           np.where(gal_dist_proj<=self.r_gal)[0])
        
        #print ("Projected z_dists:", proj_gas_z_dist[0], gal_dist_proj[0], len(proj_gas_z_dist))
        #if len(proj_gas_z_dist)==1 and proj_gas_z_dist[0]<=self.h and gal_dist_proj[0]<=self.r_gal:
        #    print ("Single particle case...", proj_gas_coords)
        #    self.data["Coordinates"] = proj_gas_coords
        #    self.data["GalDist"] = gal_dist_proj
        #else:
        self.disk_inds = np.where((gal_dist_proj<=self.r_gal)&(proj_gas_z_dist<=self.h))[0]
        #print ("Disk indices:", self.disk_inds)
        proj_gas_coords_x = np.take(proj_gas_coords[:,0], self.disk_inds)
        proj_gas_coords_y = np.take(proj_gas_coords[:,1], self.disk_inds)
        proj_gas_coords_z = np.take(proj_gas_coords[:,2], self.disk_inds)

        self.data["Coordinates"] = np.array([proj_gas_coords_x,proj_gas_coords_y,proj_gas_coords_z]).T
        self.data["GalDist"] = np.take(gal_dist_proj, self.disk_inds)

    def add_key(self, key, init_data, dim, projection=True):
        if dim==1:    
            temp_data = np.take(init_data, self.sphere_inds)
            self.data[key] = np.take(temp_data, self.disk_inds)

        if dim==3:
            temp_data_x = np.take(init_data[:,0], self.sphere_inds)
            temp_data_y = np.take(init_data[:,1], self.sphere_inds)
            temp_data_z = np.take(init_data[:,2], self.sphere_inds)
            
            final_data_x = np.take(temp_data_x, self.disk_inds)
            final_data_y = np.take(temp_data_y, self.disk_inds)
            final_data_z = np.take(temp_data_z, self.disk_inds)
            final_data = np.array([final_data_x, final_data_y, final_data_z]).T
            #self.data[key] = np.array([final_data_x, final_data_y, final_data_z]).T
            if projection:
                if self.verbose:
                    print ("Starting projection...")

                proj_final_data = []
                for i in range(0, len(final_data)):
                    proj_final_datum = np.matmul(self.proj, final_data[i])
                    proj_final_data.append(proj_final_datum)
                    #if i%5000==0:
                        #print('Percent complete:', i/len(gas_coords)*100)
                
                self.data[key] = np.array(proj_final_data)
                if self.verbose:
                    print ("Projection done...")
            
            else:
                self.data[key] = final_data
    
    def add_array(self, key, data):
        self.data[key] = data



def load_gal_quants(params, snapnum, particle_type=0):
    gal_quants0 = np.load(params.path+params.gal_quants_sub_dir+f'gal_quants{particle_type}_snap'+str(snapnum)+'.npy', allow_pickle=True).item()
    return gal_quants0

def load_vel_sub(snapnum, params, multiple=1):
    if multiple == 1:
        vel_sub = np.load(params.path+params.vels_sub_dir+"vels_sub_1x_snap"+str(snapnum)+".npy")
    elif multiple == 5:
        vel_sub = np.load(params.path+params.vels_sub_dir+"vels_sub_5x_snap"+str(snapnum)+".npy")
    elif multiple == 0.1:
        vel_sub = np.load(params.path+params.vels_sub_dir+"vels_sub_0_1x_snap"+str(snapnum)+".npy")
    elif multiple == 0.5:
        vel_sub = np.load(params.path+params.vels_sub_dir+"vels_sub_0_5x_snap"+str(snapnum)+".npy")
    else:
        print ("Invalid multiple")
    return vel_sub


def get_vc(gal_quants0, gal_quants1, gal_quants4, r_gals):
    """
    Function to calculat the circular velocity of the galaxy as a function of galactocentric distance.

    Parameters
    ----------
    gal_quants0 : GalQuants
        The class containing the gas data for the galaxy.
    gal_quants1 : GalQuants
        The class containing the DM data for the galaxy.
    gal_quants4 : GalQuants
        The class containing the star data for the galaxy.
    r_gals : array_like
        The galactocentric distances at which to calculate the circular velocity.

    Returns
    -------
    vc : array_like
        The circular velocity of the galaxy as a function of galactocentric distance.
    """
    vc = []
    for r_gal in r_gals:
        #m0 = np.take(masses0, np.where((gal_dist0<=r_gal)&\
        #                               (np.abs(positions0[:,2]-gal_centre[2])<=))[0])
        m0 = np.take(gal_quants0.data["Masses"], np.where(gal_quants0.data["GalDist"]<=r_gal)[0])
        m1 = np.take(gal_quants1.data["Masses"], np.where(gal_quants1.data["GalDist"]<=r_gal)[0])
        #m2 = np.take(gal_quants2.data["Masses"], np.where(gal_quants2.data["GalDist"]<=r_gal)[0])
        m4 = np.take(gal_quants4.data["Masses"], np.where(gal_quants4.data["GalDist"]<=r_gal)[0])

        m_tot = np.sum(m0)+np.sum(m1)+np.sum(m4) #np.sum(m2)+
        vc.append(np.sqrt(const.G*const.Msun/const.kpc**3*const.Myr**2*m_tot*1e10/r_gal))                #G is in Msun-1 Myr^-2 kpc^3. 

    vc = np.array(vc)*const.kpc/const.Myr/1e5     #in km/s

    return vc
