"""
This file contains utility functions for GUACAMOLE scripts. 

Author: Shivan Khullar
Date: June 2024
"""
import numpy as np

def convert_to_array(string, dtype=np.int32):
    """
    Convert a string of comma separated integers to a numpy array.
    """
    li = list(string.split(","))
    return np.array(li).astype(dtype)

def convert_to_bool(string):
    """
    Convert a string to a boolean.
    """
    if string.lower() in ['true', 't', '1', 'yes', 'True', 'T']:
        return True
    elif string.lower() in ['false', 'f', '0', 'no', 'False', 'F']:
        return False
    else:
        raise ValueError("Invalid boolean string.")

