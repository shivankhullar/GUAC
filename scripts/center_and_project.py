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
#matplotlib.use('Agg')
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
    snapnum = int(args['--snapnum'])
    #r_gal = float(args['--r_gal'])
    #h = float(args['--h'])
    save_path = path+args['--save_path']
    
    #image_path = path+'img_data/'

    print ("Finding center and projection matrix...")
    make_photo(path, snapnum, save_path)
    #image_path = save_path
        
