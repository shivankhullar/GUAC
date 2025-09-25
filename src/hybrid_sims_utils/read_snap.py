"""
This file contains some general utility functions for hybrid STARFORGE+FIRE simulations.

Author: Shivan Khullar
Date: Feb 2025

"""
import h5py
#from galaxy_utils.gal_utils import *



def get_snap_data_hybrid(sim, sim_path, snap, snapshot_suffix='', snapdir=True, refinement_tag=False, verbose=True):

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
    if refinement_tag:
        for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber", "RefinementFlag": #, "MagneticField", "Potential":
            pdata[field] = F["PartType0"][field][:]#[density_cut]
    else:
        for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDs", "ParticleIDGenerationNumber", "MagneticField":
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