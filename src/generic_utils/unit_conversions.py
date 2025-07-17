"""
This file contains utility functions for converting between cosmological units and physical. 

Author: Shivan Khullar
Date: July 2024
"""
import numpy as np


def convert_to_physical(x, h, a, key, length_dim=0, mass_dim=0, vel_dim=0):
    """
    Convert a quantity from cosmological to physical units.

    Parameters
    ----------
    x : array of floats
        The quantity to convert.
    h : float
        The Hubble constant.
    a : float
        The scale factor.
    length_dim : int, optional
        The power of length (dimensionality). Default is 0.
    mass_dim : int, optional
        The power of mass (dimensionality). Default is 0.
    vel_dim : int, optional
        The power of velocity (dimensionality). Default is 0.
    
    Returns
    -------
    quant: array of floats
        The converted quantity.

    See (https://arepo-code.org/wp-content/userguide/snapshotformat.html)
    """
    length_factor = a/h
    mass_factor = 1/h
    vel_factor = np.sqrt(a)
    if key == 'coords' or key=='Coords'\
        or key=='coordinates' or key=='Coordinates'\
            or key == 'position' or key=='Position'\
                or key == 'length' or key=='Length' \
                    or key=='c' or key=='C':
        quant = x * length_factor 
    elif key == 'velocity' or key=='Velocity'\
        or key=='vels' or key=='Vels'\
            or key=='velocities' or key=='Velocities'\
                or key=='v' or key=='V':
        vel_dim = 1
    elif key == 'mass' or key=='Mass'\
        or key=='masses' or key=='Masses'\
            or key=='m' or key=='M':
        mass_dim = 1
    elif key=='density' or key=='Density'\
        or key=='densities' or key=='Densities'\
            or key=='rho' or key=='Rho' or key=='d' or key=='D':
        mass_dim = 1
        length_dim = -3
    
    #return
    return quant