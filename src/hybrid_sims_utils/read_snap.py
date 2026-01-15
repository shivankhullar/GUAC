"""
This file contains some general utility functions for hybrid STARFORGE+FIRE simulations.

Author: Shivan Khullar
Date: Feb 2025

"""
import h5py
import numpy as np
#from galaxy_utils.gal_utils import *



def get_snap_data_hybrid(sim, sim_path, snap, snapshot_suffix='', snapdir=True, refinement_tag=False, verbose=True, full_tag=False, movie_tag=False):

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
    pdata = {}
    if full_tag:
        for field in F['PartType0'].keys():
            pdata[field] = F["PartType0"][field][:]
    elif refinement_tag:
        for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber", "RefinementFlag": #, "MagneticField", "Potential":
            pdata[field] = F["PartType0"][field][:]#[density_cut]
    elif movie_tag:
        for field in "Masses", "Coordinates", "SmoothingLength", "Velocities", "Temperature": #, "MagneticField", "Potential":
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

    for key in F['Header'].attrs.keys():
        pdata[key] = F['Header'].attrs[key]
    
    stardata = {}
    if 'PartType5' in F.keys():
        try:
            #for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber", "StellarFormationTime":
            #    stardata[field] = F["PartType5"][field][:]#[density_cut]
            for field in F['PartType5'].keys():
                stardata[field] = F["PartType5"][field][:]

            for key in F['Header'].attrs.keys():
                stardata[key] = F['Header'].attrs[key]
        except:
            print('No STARFORGE stars data in this snapshot')

    fire_stardata = {}
    if 'PartType4' in F.keys():
        try:
            for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber":
                fire_stardata[field] = F["PartType4"][field][:]#[density_cut]

            for key in F['Header'].attrs.keys():
                fire_stardata[key] = F['Header'].attrs[key]
        except:
            print('No FIRE stars data in this snapshot')
    refine_data = {}
    if 'PartType3' in F.keys():
        for field in "Masses", "Coordinates", "Velocities":
            refine_data[field] = F["PartType3"][field][:]#[density_cut]

    #refine_pos = np.array(F['PartType3/Coordinates'])
    #refine_pos = refine_pos[0]
    F.close()

    return pdata, stardata, fire_stardata, refine_data, snapname




def convert_units_to_physical(pdata, stardata, fire_stardata):
    if pdata:
        a = pdata['Time']
        h = pdata['HubbleParam']
        hinv = 1.0 / h
        rconv = a * hinv 

    data_dicts = [pdata, stardata, fire_stardata]
    for data_dict in data_dicts: 
        for key in data_dict.keys():
            if key == "Coordinates" or key == "SmoothingLength":
                data_dict[key] = data_dict[key] * rconv
            if key == "RefinementRegionCenter":
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
        

    return pdata, stardata, fire_stardata