"""
This file contains utility functions for the FIRE simulations. It includes functions to load data from the FIRE
simulations, get the galaxy centre and projection matrix, and get the temperature of the gas.

Author: Shivan Khullar
Date: April 2024
"""

import h5py

#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import load_from_snapshot
from generic_utils.constants import *
from galaxy_utils.gal_utils import *
import yt 
from yt.units import parsec, Msun
from yt.utilities.cosmology import Cosmology


class Params():
    """
    This class is used to store parameters for the FIRE simulations. Eveything is built around this class
    so that we can easily change parameters and have them propagate through the code.

    Inputs:
    path: string, path to the simulation data
    nmin: int, minimum number density for the cloud identification
    vir: int, virial parameter for cloud identification
    sub_dir: string, subdirectory for the cloud tracking data 
    start_snap: int, first snapshot number
    last_snap: int, last snapshot number
    filename_prefix: string, prefix for the linked cloud files
    cloud_num_digits: int, number of digits in the cloud number
    snapshot_num_digits: int, number of digits in the snapshot number
    cloud_prefix: string, prefix for the cloud files
    snapshot_prefix: string, prefix for the snapshot files
    age_cut: float, age cut for star particles
    dat_file_header_size: int, header size for the linked cloud files
    star_data_sub_dir: string, subdirectory for the star data
    cph_sub_dir: string, subdirectory for the cloud phinder data
    image_path: string, path to the image data
    img_fname_pre: string, prefix for the image files
    img_fname_suf: string, suffix for the image files
    hdf5_file_prefix: string, prefix for the hdf5 files
    frac_thresh: string, threshold for the cloud phinder data
    gas_data_sub_dir: string, subdirectory for the gas data
    sim: string, simulation name
    r_gal: float, galaxy radius
    h: float, scale height
    """
    def __init__(self, path = "../../FIRE/m12i_res7100/", nmin = 10, vir = 2, \
                    sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=10, vir=2), \
                    start_snap = 590, last_snap = 600, filename_prefix="Linked_Clouds_", cloud_num_digits = 4, \
                    snapshot_num_digits = 3, cloud_prefix = "Cloud", snapshot_prefix="Snap", age_cut=1, \
                    dat_file_header_size=8, gas_data_sub_dir = 'GasData/', star_data_sub_dir = "StarData/", \
                    cph_sub_dir="CloudPhinderData/", image_path="img_data/", \
                    img_fname_prefix='center_proj_', img_fname_suffix='.hdf5', \
                    hdf5_file_prefix = 'Clouds_', frac_thresh='thresh0.0', sim='m12i_final_fb_57k', \
                    r_gal=25, h=0.4,  gal_quants_sub_dir='gal_quants/', vels_sub_dir='vel_sub/', phinder_sub_dir="VoidPhinderData/"):

        self.sim=sim
        self.path=path+sim+'/'
        self.nmin=nmin
        self.vir=vir
        self.sub_dir = sub_dir
        self.start_snap = start_snap
        self.last_snap = last_snap
        self.filename_prefix = filename_prefix#+"n{nmin}_alpha{vir}_".format(nmin=nmin, vir=vir)
        self.cloud_num_digits=cloud_num_digits
        self.snapshot_num_digits=snapshot_num_digits
        self.cloud_prefix=cloud_prefix
        self.snapshot_prefix=snapshot_prefix
        self.age_cut=age_cut
        self.dat_file_header_size=dat_file_header_size
        self.star_data_sub_dir=star_data_sub_dir
        self.cph_sub_dir=cph_sub_dir+"n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
        self.image_path=self.path+image_path
        self.image_filename_prefix=img_fname_prefix
        self.image_filename_suffix=img_fname_suffix
        self.hdf5_file_prefix = hdf5_file_prefix
        self.frac_thresh = frac_thresh
        self.gas_data_sub_dir = gas_data_sub_dir
        self.r_gal = r_gal
        self.h = h
        self.gal_quants_sub_dir = gal_quants_sub_dir
        self.vels_sub_dir = vels_sub_dir
        self.phinder_sub_dir = phinder_sub_dir
  

def load_fire_data_arr(key, field, snapnum, params):
    """
    Load data from array files that we've previously saved from FIRE snapshots.

    Inputs:
    key: string, 'gas' or 'star'
    field: string, 'Coordinates', 'Masses', 'Velocities', 'hsml', 'Temps', 'Density' for gas
           'Coordinates', 'Masses', 'Ages' for star
    snapnum: int, snapshot number
    params: object, parameters object
    """
    if key=='gas' or key=='Gas' or key=='GasData':
        if field=='Coordinates' or field=='coords' or field=='Coords':
            arr = np.load(params.path+'GasData/'+'Coords_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']
        if field=='Masses' or field=='masses':
            arr = np.load(params.path+'GasData/'+'Masses_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']
        if field=='Velocities' or field=='vels' or field=='Vels':
            arr = np.load(params.path+'GasData/'+'Vels_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']
        if field=='hsml' or field=='Hsml' or field=='SmoothingLength':
            arr = np.load(params.path+'GasData/'+'SmoothingLength_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']
        if field=='Temps' or field=='Temp' or field=='temp' or field=='temps' or field=='Temperature' or field=='temperature':
            arr = np.load(params.path+'GasData/'+'Temps_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']
        if field=='Density' or field=='Dens' or field=='dens' or field=='density':
            arr = np.load(params.path+'GasData/'+'Dens_{snap}.npz'.format(snap=snapnum))
            data = arr['arr_0']

    if key=='star' or key=='Star' or key=='StarData':
        if field=='Coordinates' or field=='coords' or field=='Coords':
            arr = np.load(params.path+'StarData/'+'Coords_cut{cut}_{snap}.npz'.format(cut=params.age_cut, snap=snapnum))
            data_x = arr['arr_0']
            data_y = arr['arr_1']
            data_z = arr['arr_2']
            data = np.column_stack((data_x, data_y, data_z))
        if field=='Masses' or field=='masses':
            arr = np.load(params.path+'StarData/'+'Masses_cut{cut}_{snap}.npz'.format(cut=params.age_cut, snap=snapnum))
            data = arr['arr_0']
        if field=='Ages' or field=='ages':
            arr = np.load(params.path+'StarData/'+'Ages_cut{cut}_{snap}.npz'.format(cut=params.age_cut, snap=snapnum))
            data = arr['arr_0']

    return data



def get_temperature(internal_egy_code, helium_mass_fraction, electron_abundance, total_metallicity, mass_density, f_neutral=np.zeros(0), f_molec=np.zeros(0), key='Temperature'):
    ''' This function is taken from Phil Hopkins' repo pfh_python. Here's a description:
        Return estimated gas temperature, given code-units internal energy, helium
        mass fraction, and electron abundance (number of free electrons per H nucleus).
        this will use a simply approximation to the iterative in-code solution for
        molecular gas, total metallicity, etc. so does not perfectly agree
        (but it is a few lines, as opposed to hundreds) 
        
        Inputs: 
        internal_egy_code: internal energy in code units
        helium_mass_fraction: helium mass fraction
        electron_abundance: number of free electrons per H nucleus
        total_metallicity: total metallicity
        mass_density: mass density
        f_neutral: neutral fraction
        f_molec: molecular fraction
        key: string, 'Temperature', 'Weight', 'Number'
        
        Output: gas temperature in Kelvin, weight, or number    
    '''
    internal_egy_cgs=internal_egy_code*1.e10; gamma_EOS=5./3.; kB=1.38e-16; m_proton=1.67e-24; X0=0.76;
    total_metallicity[(total_metallicity>0.25)] = 0.25;
    helium_mass_fraction[(helium_mass_fraction>0.35)] = 0.35;
    y_helium = helium_mass_fraction / (4.*(1.-helium_mass_fraction));
    X_hydrogen = 1. - (helium_mass_fraction+total_metallicity);
    T_mol = 100. * (mass_density*4.04621);
    T_mol[(T_mol>8000.)] = 8000.;
    A0 = m_proton * (gamma_EOS-1.) * internal_egy_cgs / kB;
    mu = (1. + 4.*y_helium) / (1.+y_helium+electron_abundance);
    X=X_hydrogen; Y=helium_mass_fraction; Z=total_metallicity; nel=electron_abundance;

    if(np.array(f_molec).size > 0):
        fmol = 1.*f_molec;
        fH=X_hydrogen; f=fmol; xe=nel;
        gamma_eff = 1. + (fH*((1.-f)/1. + f/2.) + (1.-fH)/4.) / (fH*((1.-f + xe)/(1.*(5./3.-1.)) + f/(2.*(7./5.-1.))) + (1.-fH)/(4.*(5./3.-1.))); ## assume He is atomic, H has a mass fra$
        A0 = m_proton * (gamma_eff-1.) * internal_egy_cgs / kB;
    else:
        fmol=0;
        if(2==2):
            for i in range(3):
                mu = 1. / ( X*(1.-0.5*fmol) + Y/4. + nel*X0 + Z/(16.+12.*fmol));
                T = mu * A0 / T_mol;
                fmol = 1. / (1. + T*T);
    mu = 1. / ( X*(1.-0.5*fmol) + Y/4. + nel*X0 + Z/(16.+12.*fmol));
    T = mu * A0;
    if('Temp' in key or 'temp' in key): return T;
    if('Weight' in key or 'weight' in key): return mu;
    if('Number' in key or 'number' in key): return mass_density * 404.621 / mu;
    return T;



def get_galaxy_centre(params, snap_num):
    """
    Get the galaxy centre from the image data file (center_and_proj_(snapnum).hdf5)

    Inputs:
    params: object, parameters object
    snap_num: int, snapshot number

    Output:
    centre: array, galaxy centre
    """
    fname = params.image_path+params.image_filename_prefix+str(snap_num)+params.image_filename_suffix
    f = h5py.File(fname,'r')
    #f.keys()
    centre = np.array(f['centering'])
    return centre

def get_galaxy_proj_matrix(params, snap_num):
    """
    Get the galaxy projection matrix from the image data file (center_and_proj_(snapnum).hdf5)

    Inputs:
    params: object, parameters object
    snap_num: int, snapshot number

    Output:
    proj: array (3x3), projection matrix
    """
    fname = params.image_path+params.image_filename_prefix+str(snap_num)+params.image_filename_suffix
    f = h5py.File(fname,'r')
    #f.keys()
    proj = f['ProjectionMatrix']
    proj = np.array(proj)
    return proj


def load_fire_data(dataset, ptype, snapdir, snapnum):
    """
    Wrapper around load_from_snapshot.load_from_snapshot to load data from the FIRE simulations.

    Inputs:
    dataset: string, dataset name
    ptype: int, particle type
    snapdir: string, snapshot directory
    snapnum: int, snapshot number

    Output:
    data: array, data from the snapshot
    """
    data = load_from_snapshot.load_from_snapshot(dataset, ptype, snapdir, snapnum, units_to_physical=True)
    return data


   
def get_snap_name(snap_num):
    """
    Get the snapshot name for the given snapshot number.

    Inputs:
    snap_num: int, snapshot number
    params: object, parameters object

    Output:
    name: string, snapshot name
    """
    if snap_num<10:
        name = "00"+str(snap_num)
    if snap_num>=10 and snap_num<100:
        name = "0"+str(snap_num)
    if snap_num>=100 and snap_num<1000:
        name = str(snap_num)
    else:
        name = str(snap_num)
    return name



def get_cloud_name(cloud_num, params):
    """
    Get the cloud name for the given cloud number.

    Inputs:
    cloud_num: int, cloud number
    params: object, parameters object

    Output:
    name: string, cloud name
    """
    if cloud_num<10:
        name = params.cloud_prefix+"000"+str(cloud_num)
        return name
    if cloud_num>=10 and cloud_num<100:
        name = params.cloud_prefix+"00"+str(cloud_num)
        return name
    if cloud_num>=100 and cloud_num<1000:
        name = params.cloud_prefix+"0"+str(cloud_num)
        return name
    else:
        name = params.cloud_prefix+str(cloud_num)
        return name
    return None
    



def read_scale_factor_from_txtfile(path, line_number):
    path = path + '/snapshot_scale_factors.txt'
    with open(path, 'r') as file:
        lines = file.readlines()
    if 0 <= line_number <= len(lines):
        value = float(lines[line_number].strip())  
        return value
    else:
        raise IndexError("Line number out of range.")

    


def get_cosomology(hubble_constant=0.702, omega_matter=0.272, omega_lambda=0.728, omega_radiation=0.0, omega_curvature=0.0):
    """
    Returns the default cosmology used by FIRE.
    """
    # FIRE defaults.
    co = Cosmology(hubble_constant, omega_matter, omega_lambda, omega_radiation, omega_curvature)
    return co



def get_snap_data(params, snap_num, gal_quants=True, cosmo=False):
    """
    Load the snapshot data for the given snapshot number.

    Parameters
    ----------
    params : Params object
        Object containing the simulation parameters.
    snap_num : int

    Returns
    -------
    snap_data : dictionary containing the snapshot data
    """
    snap_data = dict()
    print ('Loading snap data for snapshot:', snap_num, params.sim)

    if gal_quants==True:
        gal_quants0 = load_gal_quants(params, snap_num, 0)
        

        snap_data['dens'] = gal_quants0.data["Density"]
        snap_data['coords'] = gal_quants0.data["Coordinates"]
        snap_data['masses'] = gal_quants0.data["Masses"]
        snap_data['temps'] = gal_quants0.data["Temperature"]
        if gal_quants0.data["SoundSpeed"] is not None:
            snap_data['cs'] = gal_quants0.data["SoundSpeed"]
        else:
            print ("Sound speed not found in gal_quants0, calculating from temperature...")
            weights = get_temperature(gal_quants0.data["InternalEnergy"], z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key='Weight')
            m_p = 1.67e-24          # mass of proton (g)
            k_B = 1.38e-16          # Boltzmann constant (erg/K)
            cs = np.sqrt(k_B*gal_quants0.data["Temperature"]/(weights*m_p))/(1e5)  # in km/s
            snap_data['cs'] = cs
        snap_data['vels'] = gal_quants0.data["Velocities"]
        snap_data['hsml'] = gal_quants0.data["SmoothingLength"]
        snap_data['pIDs'] = gal_quants0.data["ParticleIDs"]
        snap_data['pIDgennum'] = gal_quants0.data["ParticleIDGenerationNumber"]
        snap_data['pIDchilds'] = gal_quants0.data["ParticleChildIDsNumber"]
        snap_data['int_energy'] = gal_quants0.data["InternalEnergy"]


        #snap_data['vels'] = vels
        #snap_data['hsml'] = hsml
        #snap_data['pIDs'] = pIDs
        #snap_data['int_energy'] = int_energy
        #if cosmo==True:
        #    snap_data['time'] = time
        #else:
        

        gal_quants4 = load_gal_quants(params, snap_num, 4)
        snap_data['star_coords'] = gal_quants4.data["Coordinates"]
        snap_data['star_masses'] = gal_quants4.data["Masses"]
        snap_data['star_vels'] = gal_quants4.data["Velocities"]
        snap_data['sfts'] = gal_quants4.data["StellarFormationTime"]
        snap_data['star_pIDs'] = gal_quants4.data["ParticleIDs"]
        snap_data['star_pIDgennum'] = gal_quants4.data["ParticleIDGenerationNumber"]
        snap_data['star_pIDchilds'] = gal_quants4.data["ParticleChildIDsNumber"]
        
        # Find ages of stars
        a_time = read_scale_factor_from_txtfile(params.path, snap_num-params.start_snap)
        co = get_cosomology()
        time = float(co.t_from_a(a_time))/Myr  # in Myr
        snap_data['time'] = time
        ages = time - np.array(co.t_from_a(gal_quants4.data["StellarFormationTime"])/Myr)
        #gal_quants4.add_array("StellarAges", ages)
        snap_data['star_ages'] = ages
        return snap_data






    if cosmo==True:
        print ('All data being returned is in cosmological units.')
        snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
        dens = load_from_snapshot.load_from_snapshot('Density', 0, snapdir, snap_num, units_to_physical=False)
        coords = load_from_snapshot.load_from_snapshot('Coordinates', 0, snapdir, snap_num, units_to_physical=False)
        masses = load_from_snapshot.load_from_snapshot('Masses', 0, snapdir, snap_num, units_to_physical=False)
        vels = load_from_snapshot.load_from_snapshot('Velocities', 0, snapdir, snap_num, units_to_physical=False)
        hsml = load_from_snapshot.load_from_snapshot('SmoothingLength', 0, snapdir, snap_num, units_to_physical=False)
        int_energy = load_from_snapshot.load_from_snapshot('InternalEnergy', 0, snapdir, snap_num, units_to_physical=False)
        metal = load_from_snapshot.load_from_snapshot('Metallicity', 0, snapdir, snap_num, units_to_physical=False)
        n_elec = load_from_snapshot.load_from_snapshot('ElectronAbundance', 0, snapdir, snap_num, units_to_physical=False)
        pIDs = load_from_snapshot.load_from_snapshot('ParticleIDs', 0, snapdir, snap_num, units_to_physical=False)
        temps = None
        F = h5py.File(snapdir+'snapshot_{num}.0.hdf5'.format(num=snap_num), 'r')
        time = F['Header'].attrs['Time']

    
    else:        
        try:
            dens = load_fire_data_arr('gas', 'dens', snap_num, params)
        except:
            dens = load_fire_data('Density', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            print('Loaded densities...')
        try:
            coords = load_fire_data_arr('gas', 'coords', snap_num, params)
        except:
            coords = load_fire_data('Coordinates', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            print('Loaded coords...')
        try:
            masses = load_fire_data_arr('gas', 'masses', snap_num, params)
        except:
            masses = load_fire_data('Masses', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            print('Loaded masses...')
        try:
            vels = load_fire_data_arr('gas', 'vels', snap_num, params)
        except:
            vels = load_fire_data('Velocities', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            print('Loaded vels...')
        try:
            hsml = load_fire_data_arr('gas', 'hsml', snap_num, params)
        except:
            hsml = load_fire_data('SmoothingLength', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
            print('Loaded hsmls...')

        int_energy = load_fire_data('InternalEnergy', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
        pIDs = load_fire_data('ParticleIDs', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)

        try:
            temps = load_fire_data_arr('gas', 'temps', snap_num, params)
        except:
            temps = None
        metal = load_fire_data('Metallicity', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
        n_elec = load_fire_data('ElectronAbundance', 0, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
        
    f_neutral = np.zeros(0)
    f_molec = np.zeros(0)
    z_tot = metal[:, 0]
    z_he = metal[:, 1]

    value='Temp'
    if temps is None:
        temps = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key=value)
    weights = get_temperature(int_energy, z_he, n_elec, z_tot, dens, f_neutral=f_neutral, f_molec=f_molec, key='Weight')

    m_p = 1.67e-24          # mass of proton (g)
    k_B = 1.38e-16          # Boltzmann constant (erg/K)
    cs = np.sqrt(k_B*temps/(weights*m_p))/(1e5)  # in km/s

    snap_data['dens'] = dens
    snap_data['coords'] = coords
    snap_data['masses'] = masses
    snap_data['temps'] = temps
    snap_data['cs'] = cs
    snap_data['vels'] = vels
    snap_data['hsml'] = hsml
    snap_data['pIDs'] = pIDs
    snap_data['int_energy'] = int_energy
    if cosmo==True:
        snap_data['time'] = time
    else:
        snap_data['time'] = 0


    # Now do the same for star data
    if cosmo==True:
        star_coords = load_from_snapshot.load_from_snapshot('Coordinates', 4, snapdir, snap_num, units_to_physical=False)
        star_masses = load_from_snapshot.load_from_snapshot('Masses', 4, snapdir, snap_num, units_to_physical=False)
        star_vels = load_from_snapshot.load_from_snapshot('Velocities', 4, snapdir, snap_num, units_to_physical=False)
        sfts = load_from_snapshot.load_from_snapshot('StellarFormationTime', 4, snapdir, snap_num, units_to_physical=False)
        star_pIDs = load_from_snapshot.load_from_snapshot('ParticleIDs', 4, snapdir, snap_num, units_to_physical=False)
    else:
        star_coords = load_fire_data_arr('star', 'coords', snap_num, params)
        star_masses = load_fire_data_arr('star', 'masses', snap_num, params)
        star_ages = load_fire_data_arr('star', 'ages', snap_num, params)
        star_pIDs = load_fire_data('ParticleIDs', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
        star_vels = load_fire_data('Velocities', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)
        sfts = load_fire_data('StellarFormationTime', 4, params.path+'snapdir_{num}'.format(num=snap_num), snap_num)

    snap_data['star_coords'] = star_coords
    snap_data['star_masses'] = star_masses
    if not cosmo: snap_data['star_ages'] = star_ages
    snap_data['star_pIDs'] = star_pIDs
    snap_data['star_vels'] = star_vels
    snap_data['sfts'] = sfts

    
    return snap_data
