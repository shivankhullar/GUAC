#!/usr/bin/env python
"""
cosmo_time_converter: Convert between redshift, scale factor, lookback time, and age of universe

This script provides conversions between cosmological time quantities including:
- Redshift (z)
- Scale factor (a) 
- Age of universe (Myr)
- Lookback time (Myr)

Can operate on single values via command line arguments or batch process files.
"""

import numpy as np
import argparse
from yt.utilities.cosmology import Cosmology

    
import h5py
import yt 
from yt.units import parsec, Msun
from generic_utils.constants import *

# Constants
#Myr = 3.15576e13  # in seconds


def get_cosmology(hubble_constant=0.702, omega_matter=0.272, omega_lambda=0.728, 
                  omega_radiation=0.0, omega_curvature=0.0):
    """
    Returns a cosmology object with specified parameters.
    Default values are FIRE cosmology.
    """
    co = Cosmology(hubble_constant, omega_matter, omega_lambda, 
                   omega_radiation, omega_curvature)
    return co


def convert_single_value(co, redshift=None, scale_factor=None, time_myr=None):
    """
    Convert between redshift, scale factor, and time for a single value.
    
    Parameters:
    -----------
    co : Cosmology object
    redshift : float, optional
        Input redshift
    scale_factor : float, optional  
        Input scale factor
    time_myr : float, optional
        Input time in Myr
        
    Returns:
    --------
    dict with all cosmological quantities
    """
    
    # Determine which input was provided and convert to scale factor
    if redshift is not None:
        a = 1.0 / (1.0 + redshift)
        z = redshift
    elif scale_factor is not None:
        a = scale_factor
        z = 1.0/a - 1.0
    elif time_myr is not None:
        # Convert time to scale factor
        time_sec = time_myr * Myr
        a = float(co.a_from_t(time_sec))
        z = 1.0/a - 1.0
    else:
        raise ValueError("Must provide either redshift, scale_factor, or time_myr")
    
    # Calculate all quantities
    time_sec = float(co.t_from_a(a))
    age_myr = time_sec / Myr
    
    # Calculate lookback time (current age - age at redshift z)
    current_time_sec = float(co.t_from_a(1.0))  # Age at z=0
    current_age_myr = current_time_sec / Myr
    lookback_time_myr = current_age_myr - age_myr
    
    return {
        'redshift': round(z, 6),
        'scale_factor': round(a, 6), 
        'age_myr': round(age_myr, 3),
        'lookback_time_myr': round(lookback_time_myr, 3),
        'current_age_myr': round(current_age_myr, 3)
    }


def print_conversion_result(result):
    """Print formatted conversion results."""
    print(f"Redshift (z):              {result['redshift']}")
    print(f"Scale factor (a):          {result['scale_factor']}")
    print(f"Age of universe:           {result['age_myr']:.3f} Myr")
    print(f"Lookback time:             {result['lookback_time_myr']:.3f} Myr")
    print(f"Current age (z=0):         {result['current_age_myr']:.3f} Myr")


def convert_file(co, filename, file_format='time', output_file=None):
    """
    Convert values from a file.
    
    Parameters:
    -----------
    co : Cosmology object
    filename : str
        Input file path
    file_format : str
        Format of input file: 'time', 'redshift', or 'scale_factor'
    output_file : str, optional
        Output file path
    """
    
    try:
        values = np.loadtxt(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return
    
    # Ensure values is a 1D array
    values = np.atleast_1d(values)
    
    results = []
    
    print(f"\nConverting {len(values)} values from {filename}")
    print(f"Input format: {file_format}")
    print("-" * 80)
    print(f"{'Input':<12} {'z':<10} {'a':<10} {'Age (Myr)':<12} {'Lookback (Myr)':<15}")
    print("-" * 80)
    
    for i, val in enumerate(values):
        try:
            if file_format == 'time':
                result = convert_single_value(co, time_myr=val)
                input_str = f"{val:.3f}"
            elif file_format == 'redshift':
                result = convert_single_value(co, redshift=val)
                input_str = f"{val:.6f}"
            elif file_format == 'scale_factor':
                result = convert_single_value(co, scale_factor=val)
                input_str = f"{val:.6f}"
            else:
                raise ValueError(f"Unknown file format: {file_format}")
            
            print(f"{input_str:<12} {result['redshift']:<10.6f} {result['scale_factor']:<10.6f} "
                  f"{result['age_myr']:<12.3f} {result['lookback_time_myr']:<15.3f}")
            
            results.append(result)
            
        except Exception as e:
            print(f"Error converting value {val}: {e}")
            continue
    
    # Write to output file if specified
    if output_file is not None:
        write_results_to_file(results, output_file, file_format)
        print(f"\nResults written to {output_file}")
    
    return results


def write_results_to_file(results, output_file, input_format):
    """Write conversion results to a file."""
    
    with open(output_file, 'w') as f:
        f.write("# Cosmological conversion results\n")
        f.write(f"# Input format: {input_format}\n")
        f.write("# Columns: redshift, scale_factor, age_myr, lookback_time_myr\n")
        
        for result in results:
            f.write(f"{result['redshift']:.6f} {result['scale_factor']:.6f} "
                   f"{result['age_myr']:.3f} {result['lookback_time_myr']:.3f}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert between redshift, scale factor, lookback time, and age of universe'
    )
    
    # Input options (mutually exclusive for single values)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--z', type=float, help='Input redshift to convert')
    input_group.add_argument('--a', type=float, help='Input scale factor to convert')
    input_group.add_argument('--t', type=float, help='Input time in Myr to convert')
    input_group.add_argument('--file', type=str, help='File containing times/redshifts/scale factors to convert')
    
    # File processing options
    parser.add_argument('--file_format', type=str, default='scale_factor', 
                       choices=['time', 'redshift', 'scale_factor'],
                       help='Format of input file (default: time)')
    parser.add_argument('--output', type=str, help='Output file for batch conversions')
    
    # Cosmology parameters
    parser.add_argument('--hubble', type=float, default=0.702, help='Hubble constant (default: 0.702)')
    parser.add_argument('--omega_m', type=float, default=0.272, help='Omega matter (default: 0.272)')
    parser.add_argument('--omega_l', type=float, default=0.728, help='Omega lambda (default: 0.728)')
    parser.add_argument('--omega_r', type=float, default=0.0, help='Omega radiation (default: 0.0)')
    parser.add_argument('--omega_k', type=float, default=0.0, help='Omega curvature (default: 0.0)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    # Set up cosmology
    co = get_cosmology(args.hubble, args.omega_m, args.omega_l, args.omega_r, args.omega_k)
    
    print(f"Using cosmology: H0={args.hubble}, Ωm={args.omega_m}, ΩΛ={args.omega_l}, Ωr={args.omega_r}, Ωk={args.omega_k}")
    print(f"Current age of universe: {float(co.t_from_a(1.0))/Myr:.3f} Myr")
    print("=" * 80)
    
    # Check if file mode
    if args.file is not None:
        convert_file(co, args.file, args.file_format, args.output)
    
    # Single value conversions
    elif args.z is not None or args.a is not None or args.t is not None:
        result = convert_single_value(co, args.z, args.a, args.t)
        print_conversion_result(result)
    
    else:
        print("Error: Must provide either --z, --a, --t, or --file")
        exit(1)
    
    print("\nDone!")

