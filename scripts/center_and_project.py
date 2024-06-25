#!/usr/bin/env python
"""
center and project: "Save the center and face-on projection matrix of the galaxy"

Usage: center_and_project.py [options]

Options:
    -h, --help                              Show this screen
    --path=<path>                           Path to the simulation directory [default: ./]
    --snapnum=<snapnum>                     Snapshot number [default: 600]
    --all_snaps_in_dir=<all_snaps_in_dir>   Whether to center and project all snapshots in the directory [default: False]
    --save_path=<save_path>                 Path to save the images [default: img_data/]
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



if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    #snapdir = args['--snapdir']
    #spacing = convert_to_array(args['--spacing'])
    save_path = path+args['--save_path']
    #snapnum = int(args['--snapnum'])
    snapnum = convert_to_array(args['--snapnum'], dtype=np.int32)
    all_snaps_in_dir = convert_to_bool(args['--all_snaps_in_dir'])

    if all_snaps_in_dir:
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
            print ('No snapshots found in the given directory.')
            # Exit if no snapshots found
            exit()
        
        print ("List of snapnums:", snap_num_list)
        print ("Finding center and projection matrix for each snapshot...")
        for snap in snap_num_list:
            print ("==============================================================")
            print ("Snapshot: ", snap)
            make_photo(path, snap, save_path)
            print ("==============================================================")

    else:
        if len(snapnum)==1:
            snapnum = snapnum[0]
            print ("Finding center and projection matrix...")
            make_photo(path, snapnum, save_path)
        
        else:
            print ("Snapnum is a range, finding center and projection matrix for each snapshot...")
            for snap in range(snapnum[0], snapnum[1]+1):
                print ("==============================================================")
                print ("Snapshot: ", snap)
                make_photo(path, snap, save_path)
                print ("==============================================================")
    #r_gal = float(args['--r_gal'])
    #h = float(args['--h'])
    
    
    #image_path = path+'img_data/'

    #print ("Finding center and projection matrix...")
    #make_photo(path, snapnum, save_path)
    #image_path = save_path
        
