#!/usr/bin/env python
"""
create_glass_box_ic.py: "Setup glassy initial conditions for a box of gas particles"

Author: Shivan Khullar
Parts of this code are taken from Mike Grudic's MakeCloud.py script.

Usage: create_glass_box_ic.py [options]

Options:
    -h, --help                  Show this screen
    --N_gas=<N_gas>             Number of gas particles [default: 100000]
    --M_gas=<M_gas>             Total mass of gas particles [default: 1000]
    --box_size=<box_size>       Box size [default: 1.0]
    --vel=<vel>                 Velocity of gas particles [default: 0.0]
    --T=<T>                     Temperature of gas particles [default: 20.0]
    --glass_path=<glass_path>   Path to glass file [default: ./glass_orig.npy]
    --file_name=<file_name>     Name of output file [default: glass_cube.hdf5]
    --out_path=<out_path>       Path to output file [default: ./]
"""


from matplotlib.colors import LogNorm
from matplotlib import colors

from docopt import docopt 
import numpy as np
import h5py

from scipy import fftpack, interpolate, ndimage
from scipy.integrate import quad, odeint, solve_bvp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def get_glass_coords(N_gas, glass_path):
    x = np.load(glass_path)
    Nx = len(x)

    while len(x)*np.pi*4/3 / 8 < N_gas:
        print("Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"%(N_gas * 8 /(4*np.pi/3), len(x)))
        x = np.concatenate([x/2 + i * np.array([0.5,0,0]) + j * np.array([0,0.5,0]) + k * np.array([0, 0, 0.5]) for i in range(2) for j in range(2) for k in range(2)])
        Nx = len(x)
    print("Glass loaded!")
    return x


def get_cube_particles(x, N_gas, L, a):
    #Select the gas particles within a cube of side 'a'
    cut_x = x[np.where(np.abs(x[:,0]) < a)]
    cut_x = cut_x[np.where(np.abs(cut_x[:,1]) < a)]
    cut_x = cut_x[np.where(np.abs(cut_x[:,2]) < a)]

    print ('We have ', len(cut_x), ' particles in the cube')
    if len(cut_x) < N_gas:
        a = a + L/len(x)**1/3*(N_gas-len(cut_x))
        print ('New a: ', a)

        cut_x = get_cube_particles(x, N_gas, L, a)

    if len(cut_x) > N_gas:
        #Remove particles from the cube until we have the right number
        cut_x = cut_x[:N_gas]
        print ('Reduced number of particles to', len(cut_x))
    return cut_x



def get_gas_props(M_gas, N_gas, box_size, vel, T, glass_path):
    dm = M_gas/N_gas
    mgas = np.repeat(dm, N_gas)

    x = get_glass_coords(N_gas, glass_path)
    Nx = len(x)
    #x = 2*(x-0.5)
    #print (x, len(x), x[:,0].max(), x[:,0].min(), x[:,1].max(), x[:,1].min(), x[:,2].max(), x[:,2].min())

    print("Computing radii...")
    r = cdist(x, [np.zeros(3)])[:,0]

    print("Done! Sorting coordinates...")
    x = x[r.argsort()]

    L = max(x[:,0].max(), x[:,1].max(), x[:,2].max())
    a = (N_gas/len(x))**(1/3)*L
        
    cut_x = get_cube_particles(x, N_gas, L, a)
    cut_x = cut_x*L/a*box_size

    vel = 0.0
    vels = np.ones((N_gas, 3))*vel

    if vel!=0:
        vels[:,0] += vel_x
        vels[:,1] += vel_y
        vels[:,2] += vel_z
    

    #All units in cgs below
    #T = 10
    k_B = 1.38064852e-16
    m_H = 1.6733e-24
    mu = 2.3  #Let's assume molecular gas
    c_s = np.sqrt(3*k_B*T/mu/m_H)    # in cm/s
    c_s = c_s/1e2   # in m/s

    int_energy = 0.5*mgas*c_s**2 #(vels[:,0]**2 + vels[:,1]**2 + vels[:,2]**2)

    gas_props = {"m":mgas, "x":cut_x, "v":vels, "u":int_energy}

    return gas_props



def get_refine_props(box_size, refine_mass):
    refine_pos = np.array([box_size/2, box_size/2, box_size/2])
    refine_mass = np.array([refine_mass])
    refine_vel = np.array([0,0,0])
    refine_dict = {"m":refine_mass, "x":refine_pos, "v":refine_vel}
    return refine_dict





def write_to_file(file, gas_props, refine_props=None):
    F=h5py.File(file, 'w')
    
    F.create_group("Header")
    F.create_group("PartType0")
    F["PartType0"].create_dataset("Masses", data=gas_props["m"])
    F["PartType0"].create_dataset("Coordinates", data=gas_props["x"])
    F["PartType0"].create_dataset("Velocities", data=gas_props["v"])
    F["PartType0"].create_dataset("ParticleIDs", data=1+np.arange(len(gas_props["m"])))
    F["PartType0"].create_dataset("InternalEnergy", data=gas_props["u"])

    if refine_props is not None:
        #get_refine_props()
        F.create_group("PartType3")
        F["PartType3"].create_dataset("Masses", data=refine_props["m"])
        F["PartType3"].create_dataset("Coordinates", data=refine_props["x"])
        F["PartType3"].create_dataset("Velocities", data=refine_props["v"])
        F["PartType3"].create_dataset("ParticleIDs", data=np.array([len(gas_props["m"])+1]))
        

    F["Header"].attrs["NumPart_ThisFile"] = [len(gas_props["m"]),0,0,(1 if refine_props is not None else 0),0,0]
    F["Header"].attrs["NumPart_Total"] = [len(gas_props["m"]),0,0,(1 if refine_props is not None else 0),0,0]
    F["Header"].attrs["BoxSize"] = box_size
    F["Header"].attrs["Time"] = 0.0
    
    F.close()
    
    return


if __name__ == "__main__":
    args = docopt(__doc__)
    N_gas = int(args['--N_gas'])
    M_gas = float(args['--M_gas'])
    box_size = float(args['--box_size'])
    vel = float(args['--vel'])
    T = float(args['--T'])
    glass_path = args['--glass_path']
    file_name = args['--file_name']
    out_path = args['--out_path']
    #repo_dir = args['--repo_dir']
    #if repo_dir[-1] != "/":
    #    repo_dir += "/"
    #systype = args['--systype']


    #N_gas = 1000000
    #M_gas = 1e3
    #box_size = 1.0
    #R = 10.0

    #glass_path = '../IC-Files/glass_orig.npy'



    #filename = 'glass_cube.hdf5'
    #out_path = './'
    file = out_path+file_name
    gas_props = get_gas_props(M_gas, N_gas, box_size, vel, T, glass_path=glass_path)
    refine_props = get_refine_props(box_size, refine_mass=1e-3)
    write_to_file(file, gas_props, refine_props)

    print("Done!")