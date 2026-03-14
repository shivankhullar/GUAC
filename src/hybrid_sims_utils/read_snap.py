"""
This file contains some general utility functions for hybrid STARFORGE+FIRE simulations.

Author: Shivan Khullar
Date: Feb 2025

"""
import h5py
import numpy as np
import yt
from generic_utils.constants import *
from yt.utilities.cosmology import Cosmology
#from yt.units import parsec, Msun


def get_snap_data_hybrid(sim, sim_path, snap, snapshot_suffix='', snapdir=True, refinement_tag=False, \
                            verbose=True, full_tag=False, movie_tag=False, custom_gas_fields=None, custom_star_fields=None,\
                                ignore_data_type=None):
    """
    Read hybrid (FIRE+STARFORGE) simulation snapshot data.

    Parameters
    ----------
    sim : str
        Simulation name/identifier.
    sim_path : str
        Path to the simulation directory.
    snap : int
        Snapshot number to read.
    snapshot_suffix : str, optional
        Suffix to append to snapshot filename (default: '').
    snapdir : bool, optional
        If True, assumes snapshots are in a 'snapshots/' subdirectory (default: True).
    refinement_tag : bool, optional
        If True, reads refinement-specific fields (default: False).
    verbose : bool, optional
        If True, prints filename being read (default: True).
    full_tag : bool, optional
        If True, reads all available PartType0 fields (default: False).
    movie_tag : bool, optional
        If True, reads movie-specific fields (default: False).
    custom_gas_fields : list of str, optional
        Custom list of PartType0 (gas) fields to read. Overrides other field selection options.
    custom_star_fields : list of str, optional
        Custom list of PartType5 (STARFORGE stars) fields to read. Overrides default star fields.

    Returns
    -------
    pdata : dict
        Dictionary containing PartType0 (gas) particle data and header attributes.
    stardata : dict
        Dictionary containing PartType5 (STARFORGE stars) particle data.
    fire_stardata : dict
        Dictionary containing PartType4 (FIRE stars) particle data.
    refine_data : dict
        Dictionary containing PartType3 (refinement) particle data.
    snapname : str
        Name of the snapshot file that was read.
    """
    if snap<10:
        snapname = 'snapshot_'+snapshot_suffix+'00{num}'.format(num=snap) 
    elif snap>=10 and snap<100:
        snapname = 'snapshot_'+snapshot_suffix+'0{num}'.format(num=snap)
    else:
        snapname = 'snapshot_'+snapshot_suffix+'{num}'.format(num=snap) 
    
    if snapdir:
        filename = sim_path+sim+'/snapshots/'+snapname+'.hdf5'        
    else:
        filename = sim_path+sim+'/'+snapname+'.hdf5'

    if verbose:
        print ("Reading file:", filename)
    
    F = h5py.File(filename,"r")
    header = {}
    for key in F['Header'].attrs.keys():
        header[key] = F['Header'].attrs[key]

    pdata = {}
    if ignore_data_type is not None:
        if "gas" in ignore_data_type or 0 in ignore_data_type:
        #print ("Ignoring gas data as specified in ignore_data_type")
            pass
    else:
        if custom_gas_fields is not None:
            for field in custom_gas_fields:
                try:
                    pdata[field] = F["PartType0"][field][:]
                except:
                    print(f'Warning: Field {field} not found in PartType0')
                    continue
        elif full_tag:
            for field in F['PartType0'].keys():
                pdata[field] = F["PartType0"][field][:]
        elif refinement_tag:
            for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber", "RefinementFlag": #, "MagneticField", "Potential":
                pdata[field] = F["PartType0"][field][:]#[density_cut]
        elif movie_tag:
            for field in "Masses", "Coordinates", "SmoothingLength", "Velocities", "Temperature","Density": #, "MagneticField", "Potential":
                try:
                    pdata[field] = F["PartType0"][field][:]#[density_cut]
                except:
                    print(f'No {field} in this snapshot')
                    continue
        else:
            for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber": #, "MagneticField":
                pdata[field] = F["PartType0"][field][:]

        try:
            pdata['RefinementFlag'] = F["PartType0"]['RefinementFlag'][:]
        except:
            pass

    
    stardata = {}
    if ignore_data_type is not None:
        if "sinks" in ignore_data_type or 5 in ignore_data_type:
            pass
    else:
        if 'PartType5' in F.keys():
            try:
                if custom_star_fields is not None:
                    for field in custom_star_fields:
                        try:
                            stardata[field] = F["PartType5"][field][:]
                        except:
                            print(f'Warning: Field {field} not found in PartType5')
                            continue
                else:
                    #for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber", "StellarFormationTime":
                    #    stardata[field] = F["PartType5"][field][:]#[density_cut]
                    for field in F['PartType5'].keys():
                        stardata[field] = F["PartType5"][field][:]

                for key in F['Header'].attrs.keys():
                    stardata[key] = F['Header'].attrs[key]
            except:
                print('No STARFORGE stars data in this snapshot')

    fire_stardata = {}
    if ignore_data_type is not None:
        if "stars" in ignore_data_type or 4 in ignore_data_type:
            pass
    else:
        if 'PartType4' in F.keys():
            try:
                for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber", "StellarFormationTime":
                    fire_stardata[field] = F["PartType4"][field][:]#[density_cut]

                for key in F['Header'].attrs.keys():
                    fire_stardata[key] = F['Header'].attrs[key]
            except:
                print('No FIRE stars data in this snapshot')


    refine_data = {}
    if ignore_data_type is not None:
        if "refine" in ignore_data_type or 3 in ignore_data_type:
            pass
    else:
        if 'PartType3' in F.keys():
            for field in "Masses", "Coordinates", "Velocities":
                refine_data[field] = F["PartType3"][field][:]#[density_cut]

    #refine_pos = np.array(F['PartType3/Coordinates'])
    #refine_pos = refine_pos[0]
    F.close()

    return header, pdata, stardata, fire_stardata, refine_data, snapname




def convert_units_to_physical(header, pdata, stardata, fire_stardata):
    """
    Convert simulation units to physical units for particle data.

    Applies appropriate conversion factors to various fields based on their physical nature,
    accounting for scale factor (a) and Hubble parameter (h) dependencies.

    Parameters
    ----------
    header : dict
        Dictionary containing simulation header information.
    pdata : dict
        Dictionary containing PartType0 (gas) particle data with 'Time' and 'HubbleParam'.
    stardata : dict
        Dictionary containing PartType5 (STARFORGE stars) particle data.
    fire_stardata : dict
        Dictionary containing PartType4 (FIRE stars) particle data.

    Returns
    -------
    pdata : dict
        Gas particle data with fields converted to physical units.
    stardata : dict
        STARFORGE star data with fields converted to physical units.
    fire_stardata : dict
        FIRE star data with fields converted to physical units.

    Notes
    -----
    Conversion factors applied:
    - Coordinates, SmoothingLength: multiplied by a/h
    - Velocities: multiplied by sqrt(a)
    - Masses: multiplied by 1/h
    - Density, Pressure: multiplied by h/(a/h)^3
    - Acceleration: multiplied by h
    """
    if header:
        a = header['Time']
        h = header['HubbleParam']
        hinv = 1.0 / h
        rconv = a * hinv 

    data_dicts = [header, pdata, stardata, fire_stardata]
    for data_dict in data_dicts: 
        for key in data_dict.keys():
            if key == "Coordinates" or key == "SmoothingLength":
                data_dict[key] = data_dict[key] * rconv
            elif key == "RefinementRegionCenter":
                data_dict[key] = data_dict[key] * rconv
            elif key == "Velocities":
                data_dict[key] = data_dict[key] * np.sqrt(a)
            elif key == "Masses" or key == "BH_Mass" or key == "CosmicRayEnergy" or key == "PhotonEnergy":
                data_dict[key] = data_dict[key] * hinv
            elif key == "Density" or key == "Pressure":
                data_dict[key] = data_dict[key] * hinv / (rconv**3)
            elif key == "Acceleration" or key == "HydroAcceleration" or key == "VelocityGradient" or key == "RadiativeAcceleration":
                data_dict[key] = data_dict[key] * h
            elif key == "DensityGradient":
                data_dict[key] = data_dict[key] * hinv / (hinv**4)
            else:
                pass
        
    return header, pdata, stardata, fire_stardata


def convert_quant_to_physical(array, key=None, a=None, h=None):
    """
    Convert a single quantity from simulation units to physical units.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing the quantity to convert.
    key : str, optional
        Name of the field being converted (determines conversion factor).
        Options: "Coordinates", "SmoothingLength", "RefinementRegionCenter",
        "Velocities", "Masses", "Density", "Pressure", "Acceleration", etc.
    a : float, optional
        Scale factor (default: 1.0).
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc (default: 0.702).

    Returns
    -------
    physical_array : numpy.ndarray
        Array converted to physical units, or None if key is unknown.

    Notes
    -----
    Returns None and prints a warning if the key is not recognized.
    """
    if a is None:
        a = 1.0
        print ("Using default value a=1.0")
    if h is None:
        h = 0.702 
        print ("Using default value h=0.702")
    hinv = 1.0 / h
    rconv = a * hinv 
    if key == "Coordinates" or key == "SmoothingLength":
        physical_array = array * rconv
    elif key == "RefinementRegionCenter":
        physical_array = array * rconv
    elif key == "Velocities":
        physical_array = array * np.sqrt(a)
    elif key == "Masses" or key == "BH_Mass" or key == "CosmicRayEnergy" or key == "PhotonEnergy":
        physical_array = array * hinv
    elif key == "Density" or key == "Pressure":
        physical_array = array * hinv / (rconv**3)
    elif key == "Acceleration" or key == "HydroAcceleration" or key == "VelocityGradient" or key == "RadiativeAcceleration":
        physical_array = array * h
    elif key == "DensityGradient":
        physical_array = array * hinv / (hinv**4)
    else:
        print ("Unknown key... check source code")
        return None
    return physical_array



def convert_quant_from_physical(array, key=None, a=None, h=None):
    """
    Convert a single quantity from physical units back to simulation units.

    Inverse operation of convert_quant_to_physical.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing the quantity in physical units to convert back.
    key : str, optional
        Name of the field being converted (determines conversion factor).
        Options: "Coordinates", "SmoothingLength", "RefinementRegionCenter",
        "Velocities", "Masses", "Density", "Pressure", "Acceleration", etc.
    a : float, optional
        Scale factor (default: 1.0).
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc (default: 0.702).

    Returns
    -------
    physical_array : numpy.ndarray
        Array converted back to simulation units, or None if key is unknown.

    Notes
    -----
    Returns None and prints a warning if the key is not recognized.
    """
    if a is None:
        a = 1.0
        print ("Using default value a=1.0")
    if h is None:
        h = 0.702 
        print ("Using default value h=0.702")
    hinv = 1.0 / h
    rconv = a * hinv
    rconv_inv = 1.0 / rconv 
    if key == "Coordinates" or key == "SmoothingLength":
        physical_array = array * rconv_inv
    elif key == "RefinementRegionCenter":
        physical_array = array * rconv_inv
    elif key == "Velocities":
        physical_array = array / np.sqrt(a)
    elif key == "Masses" or key == "BH_Mass" or key == "CosmicRayEnergy" or key == "PhotonEnergy":
        physical_array = array * h
    elif key == "Density" or key == "Pressure":
        physical_array = array * h / (rconv_inv**3)
    elif key == "Acceleration" or key == "HydroAcceleration" or key == "VelocityGradient" or key == "RadiativeAcceleration":
        physical_array = array * hinv
    elif key == "DensityGradient":
        physical_array = array * h / (h**4)
    else:
        print ("Unknown key... check source code")
        return None


    return physical_array


def convert_scale_factor_to_time(a, pdata):
    """
    Convert scale factor to time in Myr using cosmology.

    Parameters
    ----------
    a : float
        Scale factor to convert.
    pdata : dict
        Dictionary containing cosmological parameters 'Omega_Matter', 'Omega_Lambda', and 'HubbleParam'.

    Returns
    -------
    age_myr : float
        Age of the universe at the given scale factor in Myr.
    """
    omega_matter = pdata['Omega_Matter']
    omega_lambda = pdata['Omega_Lambda']
    if pdata['Omega_Radiation'] is not None:
        omega_radiation = pdata['Omega_Radiation']
    else:
        print ("Omega_Radiation not found in particle data, using default value of 0.0")
        omega_radiation = 0.0
    hubble_constant = pdata['HubbleParam']

    co = yt.utilities.cosmology.Cosmology(omega_matter=omega_matter, omega_lambda=omega_lambda, hubble_constant=hubble_constant, omega_radiation=omega_radiation)
    
    time_sec = float(co.t_from_a(a))
    time_myr = time_sec / Myr

    return time_myr


def convert_formation_times_to_ages(pdata, fire_stardata):
    """
    INCOMPLETE:
    Convert stellar formation times to ages using cosmology.

    Uses yt's cosmology utilities to compute stellar ages from formation times.

    Parameters
    ----------
    pdata : dict
        Dictionary containing gas particle data with cosmological parameters:
        'Omega_Matter', 'Omega_Lambda', 'HubbleParam'.
    fire_stardata : dict
        Dictionary containing FIRE star particle data with 'StellarFormationTime'.

    Returns
    -------
    TBD
        Function appears incomplete.

    Notes
    -----
    This function is incomplete and requires omega_radiation to be defined.
    """
    omega_matter = pdata['Omega_Matter']
    omega_lambda = pdata['Omega_Lambda']
    if pdata['Omega_Radiation'] is not None:
        omega_radiation = pdata['Omega_Radiation']
    else:
        print ("Omega_Radiation not found in particle data, using default value of 0.0")
        omega_radiation = 0.0
    hubble_constant = pdata['HubbleParam']
    sfts = fire_stardata['StellarFormationTime']


    co = yt.utilities.cosmology.Cosmology(omega_matter=omega_matter, omega_lambda=omega_lambda, hubble_constant=hubble_constant, omega_radiation=omega_radiation)
    time_sec = float(co.t_from_a(a))
    age_myr = time_sec / Myr
