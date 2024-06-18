#!/usr/bin/env python
"""
center and project: "Save the center and face-on projection matrix of the galaxy"

Usage: center_and_project.py [options]

Options:
    -h, --help                  Show this screen
    --path=<path>               Path to the simulation directory [default: ./]
    --snapnum=<snapnum>         Snapshot number [default: 600]
    --save_path=<save_path>     Path to save the images [default: img_data/]
"""

from docopt import docopt
from gal_viz_utils import *
from meshoid import Meshoid
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



def convert_to_array(string):
    li = list(string.split(","))
    return np.array(li).astype(np.int32)


if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['--path']
    #snapdir = args['--snapdir']
    #spacing = convert_to_array(args['--spacing'])
    save_path = path+args['--save_path']
    #snapnum = int(args['--snapnum'])
    snapnum = convert_to_array(args['--snapnum'])

    if len(snapnum)==1:
        snapnum = snapnum[0]
        print ("Finding center and projection matrix...")
        make_photo(path, snapnum, save_path)
    
    else:
        print ("Snapnum is a range, finding center and projection matrix for each snapshot...")
        for snap in range(snapnum[0], snapnum[1]+1):
            print ("Snapshot: ", snap)
            print ("Finding center and projection matrix...")
            make_photo(path, snap, save_path)
    #r_gal = float(args['--r_gal'])
    #h = float(args['--h'])
    
    
    #image_path = path+'img_data/'

    #print ("Finding center and projection matrix...")
    #make_photo(path, snapnum, save_path)
    #image_path = save_path
        
