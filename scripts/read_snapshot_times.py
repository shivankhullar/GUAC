#!/usr/bin/env python
"""
read_snapshot_times: "Find the scale factors/time attribute of the snapshots in a given directory."

Usage: read_snapshot_times.py [options]

Options:
    -h, --help                  Show this screen
    --snapdir=<snapdir>         Directory containing snapshots [default: ./]
    --read_co=<read_co>         Whether to read the cosmology from the snapshot [default: True]
    --anyhdf5=<anyhdf5>         Whether to read any HDF5 file in the directory [default: False]
"""


import h5py
import yt
from yt.units import parsec, Msun
import numpy as np
from docopt import docopt
import glob


def print_scale_factors(snapdir, read_co=True, anyhdf5=False):
    """
    Prints the scale factors of the snapshots in the given directory.
    """
    if anyhdf5==True:
        print ("Any hdf5 is set to:", anyhdf5)
    if read_co==False:
        print ("Read cosmology is set to:", read_co)
    snap_list = np.sort(glob.glob(snapdir+'snapshot*.hdf5'))
    if snap_list.size==0:
        snap_list = np.sort(glob.glob(snapdir+'snapdir*/snapshot*.hdf5'))
    if snap_list.size==0:
        snap_list = np.sort(glob.glob(snapdir+'snapdir_*/*.hdf5'))
    if snap_list.size==0 and anyhdf5!=True:
        print ('No snapshots found in the given directory.')
        return
    if snap_list.size==0 and anyhdf5==True:
        snap_list = np.sort(glob.glob(snapdir+'*.hdf5'))
        if snap_list.size==0:
            print ('No HDF5 files found in the given directory.')
            return
    for snap in snap_list:
        f = h5py.File(snap, 'r')
        a = f['Header'].attrs['Time']
        try: 
            redshift = f['Header'].attrs['Redshift']
        except:
            redshift = 1.0/a - 1.0
        Myr = 3.15576e13
        if read_co:
            try:
                omega_matter = f['Header'].attrs['Omega0']
                omega_lambda = f['Header'].attrs['OmegaLambda']
                hubble_constant = f['Header'].attrs['HubbleParam']
                try:
                    omega_radiation = f['Header'].attrs['OmegaRadiation']
                except:
                    omega_radiation = 0.0
            except:
                omega_matter = f['Header'].attrs['Omega_Matter']
                omega_lambda = f['Header'].attrs['Omega_Lambda']
                hubble_constant = f['Header'].attrs['HubbleParam']
                omega_radiation = f['Header'].attrs['Omega_Radiation']
            co = yt.utilities.cosmology.Cosmology(omega_matter=omega_matter, omega_lambda=omega_lambda, hubble_constant=hubble_constant, omega_radiation=omega_radiation)
        else:
            print ('Cosmology not read from snapshot. Using default cosmology.')
            co = yt.utilities.cosmology.Cosmology(omega_matter=0.272, omega_lambda=0.728, hubble_constant=0.702)
        time = float(co.t_from_a(a)/Myr)
        print (snap, a, time, redshift)
        f.close()


if __name__ == '__main__':
    if docopt(__doc__)['--read_co']=='False':
        read_co = False
    else:
        read_co = True

    if docopt(__doc__)['--anyhdf5']=='True':
        anyhdf5 = True
    else:
        anyhdf5 = False
    print_scale_factors(docopt(__doc__)['--snapdir'], read_co=read_co, anyhdf5=anyhdf5)
