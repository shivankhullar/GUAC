#!/usr/bin/env python
"""
center and project: "Save the center and face-on projection matrix of the galaxy"

Usage: center_and_project.py [options]

Options:
    -h, --help                              Show this screen
    --path=<path>                           Path to the simulation directory [default: ./]
    --snapnum=<snapnum>                     Snapshot number [default: 600]
    --snapnum_range=<snapnum_range>         Range of snapshots to process (e.g., 0,100) [default: ]
    --all_snaps_in_dir=<all_snaps_in_dir>   Whether to center and project all snapshots in the directory [default: False]
    --save_path=<save_path>                 Path to save the images [default: img_data/]
    --num_processes=<num_processes>         Number of parallel processes [default: 8]
"""

from docopt import docopt
from galaxy_utils.gal_utils import *
from generic_utils.fire_utils import *
from generic_utils.script_utils import *
import glob
#from gal_viz_utils import *
from meshoid import Meshoid
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
#from visualization import *
from visualization.image_maker import edgeon_faceon_projection
import os
from multiprocessing import Pool
from functools import partial



def make_photo(path, snapnum, save_path):
    edgeon_faceon_projection(path,   #snapshot director
                         snapnum,        #snapshot number
                         field_of_view=35,  #set the size of the image
                         #image_name_prefix='Faceon_{}'.format(snapnum),
                         faceon=True,
                         pixels=2048,
                         output_directory=save_path,
                         just_center_and_project=True)       #do a faceon image (or set faceon=False or edgeon=True to get an edge-on image)
                         #**kwargs)  #kwargs are passed to image_maker, get_center, and load_fire_snap

def process_snapshot(snap, path, save_path):
    """Wrapper function for parallel processing"""
    try:
        print("==============================================================")
        print("Processing snapshot: ", snap)
        make_photo(path, snap, save_path)
        print("Completed snapshot: ", snap)
        print("==============================================================")
        return snap, True
    except Exception as e:
        print(f"Error processing snapshot {snap}: {e}")
        return snap, False



if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    save_path = path+args['--save_path']
    snapnum = convert_to_array(args['--snapnum'], dtype=np.int32)
    snapnum_range = args['--snapnum_range']
    all_snaps_in_dir = convert_to_bool(args['--all_snaps_in_dir'])
    num_processes = int(args['--num_processes'])

    snap_num_list = None

    # Determine which snapshots to process
    if snapnum_range:
        # Parse snapnum_range (e.g., "0,100" or "12,278")
        snap_range = convert_to_array(snapnum_range, dtype=np.int32)
        if len(snap_range) == 2:
            snap_num_list = np.arange(snap_range[0], snap_range[1] + 1)
            print(f"Processing snapshot range: {snap_range[0]} to {snap_range[1]}")
        else:
            print("Invalid snapnum_range format. Use: start,end")
            exit(1)
    
    elif all_snaps_in_dir:
        # Get the list of snapshots in the directory, try a bunch of ways
        snap_list = np.sort(glob.glob(path+'snapshot*.hdf5'))
        if snap_list.size>0:
            snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            snap_list = np.sort(glob.glob(path+'snapdir*/snapshot*.0.hdf5'))
            if snap_list.size>0:
                snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.0.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            snap_list = np.sort(glob.glob(path+'snapdir_*/*.hdf5'))
            if snap_list.size>0:
                snap_num_list = np.array([int(snap.split('snapshot_')[1].split('.hdf5')[0]) for snap in snap_list])
        if snap_list.size==0:
            print('No snapshots found in the given directory.')
            exit()
        print("List of snapnums:", snap_num_list)
    
    else:
        if len(snapnum)==1:
            # Single snapshot
            snap_num_list = snapnum
        else:
            # Range provided via --snapnum
            snap_num_list = np.arange(snapnum[0], snapnum[1]+1)
            print(f"Processing snapshot range: {snapnum[0]} to {snapnum[1]}")

    # Process snapshots
    if snap_num_list is not None and len(snap_num_list) > 1:
        print(f"Processing {len(snap_num_list)} snapshots using {num_processes} parallel processes...")
        
        # Create partial function with fixed path and save_path
        process_func = partial(process_snapshot, path=path, save_path=save_path)
        
        # Use multiprocessing pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, snap_num_list)
        
        # Report results
        successful = sum(1 for _, success in results if success)
        print(f"\nCompleted: {successful}/{len(snap_num_list)} snapshots processed successfully")
    
    else:
        # Single snapshot processing
        snapnum = snap_num_list[0] if hasattr(snap_num_list, '__len__') else snap_num_list
        print("Finding center and projection matrix for single snapshot...")
        make_photo(path, snapnum, save_path)
        
