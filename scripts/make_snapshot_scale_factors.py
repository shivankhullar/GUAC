#!/usr/bin/env python
"""
snapshot_timing: "Find the scale factor at which snapshots need to be produced"

Usage: snapshot_timing.py [options]

Options:
    -h, --help                  Show this screen
    --output_dir=<output>       Output directory [default: ./]
    --output_filename=<output_filename>   Output filename [default: snapshot_scale_factors.txt]
    --snapdir=<snapdir>         Directory containing snapshots [default: ./]
    --start_a=<start_a>         Starting scale factor [default: 0.98]
    --end_a=<end_a>             Ending scale factor [default: 1.0]
    --end_time=<end_time>       Ending time in Myr [default: None]
    --spacing=<spacing>         Spacing between snapshots in Myr [default: 1.0]
    --cut_off_time=<cut_off_time>   Time after which the spacing between snapshots changes [default: 25.0]
    --key=<key>                 Whether to go backwards or forwards in time [default: backwards]
"""

import h5py
import yt 
from yt.units import parsec, Msun
import numpy as np
from docopt import docopt
G = 4.492069*10**(-3)

from yt.utilities.cosmology import Cosmology

Myr = 3.15576e13            # in seconds


def get_cosomology(hubble_constant=0.702, omega_matter=0.272, omega_lambda=0.728, omega_radiation=0.0, omega_curvature=0.0):
    """
    Returns the default cosmology used by FIRE.
    """
    # FIRE defaults.
    co = Cosmology(hubble_constant, omega_matter, omega_lambda, omega_radiation, omega_curvature)
    return co

def get_scale_factors(co, start_a, end_a, spacing_in_Myr, cut_off_time=24, key='backwards', output='./', filename='snapshot_scale_factors.txt'):
    start_time = float(co.t_from_a(start_a))              # in seconds
    end_time = float(co.t_from_a(end_a))                  # in seconds
    print ('start_time = ', start_time/Myr)
    print ('end_time = ', end_time/Myr)

    if len(spacing_in_Myr)==1:
        current_spacing_in_Myr = float(spacing_in_Myr[0])
    else:
        print ('You chose to use two different spacing_in_Myr values. \n \
               The first value will be used for snapshots before \n \
                    the cut_off_time and the second value will be used for snapshots after the cut_off_time.')
    time_list=[]
    if key=='backwards' or key=='b':
        print ('Going backwards in time')
        current_time = end_time
        a_list = [end_a]
        iteration=0
        while current_time>=start_time:
            if len(spacing_in_Myr)>1:
                if end_time - current_time >= cut_off_time*Myr:
                    current_spacing_in_Myr = spacing_in_Myr[0]
                else:
                    current_spacing_in_Myr = spacing_in_Myr[1]
                current_time = current_time - current_spacing_in_Myr*Myr
            else:
                #print (type(spacing_in_Myr), type(current_time), type(Myr))
                current_time = current_time - current_spacing_in_Myr*Myr
            #print (current_time/Myr)
            a_list.append(np.round(float(co.a_from_t(current_time)), 7))
            time_list.append(current_time/Myr)
            print (iteration, current_time/Myr, current_spacing_in_Myr, a_list[-1], time_list[-1])
            iteration+=1

        a_list = a_list[::-1]

    elif key=='forwards' or key=='f':
        print ('Going forwards in time')
        current_time = start_time
        a_list = [start_a]
        iteration=0
        while current_time<=end_time:
            if len(spacing_in_Myr)>1:
                if current_time <= start_time + cut_off_time*Myr:
                    current_spacing_in_Myr = spacing_in_Myr[0]
                else:
                    current_spacing_in_Myr = spacing_in_Myr[1]
                current_time = current_time + current_spacing_in_Myr*Myr
            else:
                current_time = current_time + spacing_in_Myr*Myr

            a_list.append(np.round(float(co.a_from_t(current_time)), 7))
            time_list.append(current_time/Myr)
            print (iteration, current_time/Myr, a_list[-1], current_spacing_in_Myr, )
            iteration+=1
    else:
        raise ValueError('key must be either "backwards" or "forwards"')
    
    write_to_file(a_list, output, filename)
    return a_list, time_list


def write_to_file(a_list, output, filename):
    f = open(output+filename, 'w')
    for a in a_list:
        f.write(str(a)+'\n')
    f.close()

def convert_to_array(string):
    li = list(string.split(","))
    return np.array(li).astype(np.float64)

if __name__ == '__main__':
    args = docopt(__doc__)
    snapdir = args['--snapdir']
    output_dir = args['--output_dir']
    start_a = float(args['--start_a'])
    end_a = float(args['--end_a'])
    spacing = convert_to_array(args['--spacing']) 
    cut_off_time = float(args['--cut_off_time'])
    key = args['--key']
    co = get_cosomology()
    
    end_time = args['--end_time']
    if end_time!='None':
        end_time = float(end_time) + float(co.t_from_a(start_a))/Myr
        end_a = np.round(float(co.a_from_t(end_time*Myr)), 7)

    
    _ = get_scale_factors(co, start_a, end_a, spacing, cut_off_time, key=key, output=output_dir, filename=args['--output_filename'])
    print ('Done!')
