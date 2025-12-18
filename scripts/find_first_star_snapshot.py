#!/usr/bin/env python
"""
find_first_star_snapshot: Find the first snapshot containing star particles (PartType4 or PartType5).

Usage: find_first_star_snapshot.py [options]

Options:
    -h, --help                  Show this screen
    --snapdir=<snapdir>         Directory containing snapshots [default: ./]
    --verbose                   Print detailed information about each snapshot
"""

import h5py
import numpy as np
from docopt import docopt
import glob
import os


def find_first_star_in_snapshot(snap_file, ptype):
    """
    Find the first star (earliest formation time) in a specific snapshot for given particle type.
    
    Args:
        snap_file (str): Path to snapshot file
        ptype (str): Particle type ('PartType4' or 'PartType5')
        
    Returns:
        dict: Information about the first star or None if not found
    """
    try:
        with h5py.File(snap_file, 'r') as f:
            if ptype not in f:
                return None
                
            # Check if StellarFormationTime exists
            if 'StellarFormationTime' not in f[ptype]:
                return None
                
            formation_times = f[ptype]['StellarFormationTime'][:]
            
            # Only consider particles that have actually formed (formation time > 0)
            valid_indices = formation_times > 0
            if not np.any(valid_indices):
                return None
                
            valid_formation_times = formation_times[valid_indices]
            
            # Find the earliest formation time
            earliest_idx_in_valid = np.argmin(valid_formation_times)
            # Map back to original array index
            original_indices = np.where(valid_indices)[0]
            earliest_idx = original_indices[earliest_idx_in_valid]
            
            earliest_formation_time = formation_times[earliest_idx]
            
            # Get mass if available
            mass = None
            if 'Masses' in f[ptype]:
                mass = f[ptype]['Masses'][earliest_idx]
            elif 'Mass' in f[ptype]:
                mass = f[ptype]['Mass'][earliest_idx]
            
            # Get coordinates if available
            coordinates = None
            if 'Coordinates' in f[ptype]:
                coordinates = f[ptype]['Coordinates'][earliest_idx]
                
            return {
                'particle_index': earliest_idx,
                'formation_time': earliest_formation_time,
                'mass': mass,
                'coordinates': coordinates,
                'total_valid_stars': np.sum(valid_indices)
            }
            
    except Exception as e:
        print(f"Error analyzing {snap_file}: {e}")
        return None


def find_first_star_snapshot(snapdir, verbose=False):
    """
    Find the first snapshot containing star particles (PartType4 or PartType5).
    
    Args:
        snapdir (str): Directory containing snapshot files
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Results containing first snapshots for each particle type
    """
    
    # Find all snapshot files in the directory
    snap_patterns = [
        snapdir + 'snapshot*.hdf5',
        snapdir + 'snapdir*/snapshot*.hdf5', 
        snapdir + 'snapdir_*/*.hdf5'
    ]
    
    snap_list = []
    for pattern in snap_patterns:
        found_snaps = glob.glob(pattern)
        if found_snaps:
            snap_list = found_snaps
            break
    
    if len(snap_list) == 0:
        print('No snapshots found in the given directory.')
        return None
    
    # Sort snapshots numerically by extracting snapshot number
    def extract_snap_num(filename):
        # Extract number from snapshot_XXX.hdf5 or snapshot_XXX.N.hdf5
        basename = os.path.basename(filename)
        try:
            # Handle both snapshot_001.hdf5 and snapshot_001.0.hdf5 formats
            if '.hdf5' in basename:
                num_part = basename.split('snapshot_')[1].split('.')[0]
                return int(num_part)
        except (IndexError, ValueError):
            return 0
        return 0
    
    snap_list.sort(key=extract_snap_num)
    
    if verbose:
        print(f"Found {len(snap_list)} snapshots in {snapdir}")
        print("Searching for first occurrence of star particles...")
    
    results = {
        'PartType4': None,
        'PartType5': None
    }
    
    # Check each snapshot for star particles
    for snap_file in snap_list:
        if verbose:
            print(f"Checking {os.path.basename(snap_file)}...")
        
        try:
            with h5py.File(snap_file, 'r') as f:
                # Get snapshot info
                try:
                    scale_factor = f['Header'].attrs['Time']
                    redshift = f['Header'].attrs.get('Redshift', 1.0/scale_factor - 1.0)
                except KeyError:
                    scale_factor = None
                    redshift = None
                
                # Check for PartType4 (typically new star particles)
                if 'PartType4' in f and results['PartType4'] is None:
                    num_part4 = len(f['PartType4/Coordinates'])
                    if num_part4 > 0:
                        results['PartType4'] = {
                            'file': snap_file,
                            'scale_factor': scale_factor,
                            'redshift': redshift,
                            'num_particles': num_part4
                        }
                        
                        if verbose:
                            print(f"  Found {num_part4} PartType4 particles!")
                
                # Check for PartType5 (typically older star particles)  
                if 'PartType5' in f and results['PartType5'] is None:
                    num_part5 = len(f['PartType5/Coordinates'])
                    if num_part5 > 0:
                        results['PartType5'] = {
                            'file': snap_file,
                            'scale_factor': scale_factor,
                            'redshift': redshift,
                            'num_particles': num_part5
                        }
                        
                        if verbose:
                            print(f"  Found {num_part5} PartType5 particles!")
                
                # If we found both types, we can stop
                if results['PartType4'] is not None and results['PartType5'] is not None:
                    break
                    
        except Exception as e:
            if verbose:
                print(f"  Error reading {snap_file}: {e}")
            continue
    
    # Now analyze the first stars in each snapshot we found
    if results['PartType4'] is not None:
        if verbose:
            print(f"Analyzing first star in PartType4 snapshot...")
        first_star4 = find_first_star_in_snapshot(results['PartType4']['file'], 'PartType4')
        results['PartType4']['first_star'] = first_star4
        
    if results['PartType5'] is not None:
        if verbose:
            print(f"Analyzing first star in PartType5 snapshot...")
        first_star5 = find_first_star_in_snapshot(results['PartType5']['file'], 'PartType5')
        results['PartType5']['first_star'] = first_star5
    
    return results


def print_results(results):
    """Print the results in a formatted way."""
    
    if results is None:
        return
    
    print("\n" + "="*60)
    print("FIRST STAR FORMATION SNAPSHOTS")
    print("="*60)
    
    # Print PartType4 results
    if results['PartType4'] is not None:
        p4 = results['PartType4']
        print(f"\nPartType4 (Star particles):")
        print(f"  First found in: {os.path.basename(p4['file'])}")
        print(f"  Number of particles: {p4['num_particles']}")
        if p4['scale_factor'] is not None:
            print(f"  Scale factor: {p4['scale_factor']:.6f}")
        if p4['redshift'] is not None:
            print(f"  Redshift: {p4['redshift']:.3f}")
        
        # Print first star details
        if 'first_star' in p4 and p4['first_star'] is not None:
            star = p4['first_star']
            print(f"  First star details:")
            print(f"    Formation time: {star['formation_time']:.6f}")
            if star['mass'] is not None:
                print(f"    Mass: {star['mass']:.6e}")
            print(f"    Particle index: {star['particle_index']}")
            if star['coordinates'] is not None:
                print(f"    Coordinates: [{star['coordinates'][0]:.3f}, {star['coordinates'][1]:.3f}, {star['coordinates'][2]:.3f}]")
            print(f"    Total stars with valid formation times: {star['total_valid_stars']}")
        else:
            print(f"  No stars with valid formation times found")
    else:
        print(f"\nPartType4: No particles found in any snapshot")
    
    # Print PartType5 results
    if results['PartType5'] is not None:
        p5 = results['PartType5']
        print(f"\nPartType5 (Sink particles):")
        print(f"  First found in: {os.path.basename(p5['file'])}")
        print(f"  Number of particles: {p5['num_particles']}")
        if p5['scale_factor'] is not None:
            print(f"  Scale factor: {p5['scale_factor']:.6f}")
        if p5['redshift'] is not None:
            print(f"  Redshift: {p5['redshift']:.3f}")
        
        # Print first star details
        if 'first_star' in p5 and p5['first_star'] is not None:
            star = p5['first_star']
            print(f"  First star details:")
            print(f"    Formation time: {star['formation_time']:.6f}")
            if star['mass'] is not None:
                print(f"    Mass: {star['mass']:.6e}")
            print(f"    Particle index: {star['particle_index']}")
            if star['coordinates'] is not None:
                print(f"    Coordinates: [{star['coordinates'][0]:.3f}, {star['coordinates'][1]:.3f}, {star['coordinates'][2]:.3f}]")
            print(f"    Total stars with valid formation times: {star['total_valid_stars']}")
        else:
            print(f"  No stars with valid formation times found")
    else:
        print(f"\nPartType5: No particles found in any snapshot")
    
    if results['PartType4'] is None and results['PartType5'] is None:
        print(f"\nNo star particles (PartType4 or PartType5) found in any snapshot.")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    args = docopt(__doc__)
    
    snapdir = args['--snapdir']
    verbose = args['--verbose']
    
    # Ensure snapdir ends with a slash
    if not snapdir.endswith('/'):
        snapdir += '/'
    
    # Check if directory exists
    if not os.path.exists(snapdir):
        print(f"Error: Directory {snapdir} does not exist.")
        exit(1)
    
    print(f"Searching for star particles in: {snapdir}")
    
    # Find first star snapshots
    results = find_first_star_snapshot(snapdir, verbose=verbose)
    
    # Print results
    print_results(results)
