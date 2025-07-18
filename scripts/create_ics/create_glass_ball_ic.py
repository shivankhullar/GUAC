#!/usr/bin/env python
"""
create_glass_ball_ic.py: "Setup glassy initial conditions for a ball of gas particles (STARFORGE IC with zero or constant velocities)"

Author: Shivan Khullar
Parts of this code are taken from Mike Grudic's MakeCloud.py script.

Usage: create_glass_box_ic.py [options]

Options:
    -h, --help                  Show this screen
    --N_gas=<N_gas>             Number of gas particles [default: 100000]
    --M_gas=<M_gas>             Total mass of gas particles [default: 1000]
    --vel=<vel>                 Velocity of gas particles [default: 0.0]
    --T=<T>                     Temperature of gas particles [default: 20.0]
    --glass_path=<glass_path>   Path to glass file [default: ./glass_orig.npy]
    --file_name=<file_name>     Name of output file [default: glass_ball.hdf5]
    --out_path=<out_path>       Path to output file [default: ./]
        
    --R=<pc>             Outer radius of the cloud in pc [default: 10.0]
    --M=<msun>           Mass of the cloud in msun [default: 2e4]
    --N=<N>              Number of gas particles [default: 2000000]
    --density_exponent=<f>   Power law exponent of the density profile [default: 0.0]
    
    --bturb=<f>          Magnetic energy as a fraction of the binding energy [default: 0.1]
    --bfixed=<f>         Magnetic field in magnitude in code units, used instead of bturb if not set to zero [default: 0]
    
    --boxsize=<f>        Simulation box size
    --derefinement       Apply radial derefinement to ambient cells outside of 3* cloud radius
    --no_diffuse_gas     Remove diffuse ISM envelope fills the rest of the box with uniform density. 
    
    --B_unit=<gauss>     Unit of magnetic field in gauss [default: 1e4]
    --length_unit=<pc>   Unit of length in pc [default: 1]
    --mass_unit=<msun>   Unit of mass in M_sun [default: 1]
    --v_unit=<m/s>       Unit of velocity in m/s [default: 1]
    
    --tmax=<N>           Maximum time to run the simulation to, in units of the freefall time [default: 5]
    --nsnap=<N>          Number of snapshots per freefall time [default: 150]
    --param_only         Just makes the parameters file, not the IC
    
    --makebox            Creates a second box IC of equivalent volume and mass to the cloud
"""
# Example:  python MakeCloud.py --M=1000 --N=1e7 --R=1.0 --localdir --param_only

import os
import numpy as np
from scipy import fftpack, interpolate
from scipy.spatial.distance import cdist
import h5py
from docopt import docopt


def get_glass_coords(N_gas, glass_path):
    x = np.load(glass_path)
    Nx = len(x)

    while len(x) * np.pi * 4 / 3 / 8 < N_gas:
        print(
            "Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"
            % (N_gas * 8 / (4 * np.pi / 3), len(x))
        )
        x = np.concatenate(
            [
                x / 2
                + i * np.array([0.5, 0, 0])
                + j * np.array([0, 0.5, 0])
                + k * np.array([0, 0, 0.5])
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )
        Nx = len(x)
    print("Glass loaded!")
    return x



arguments = docopt(__doc__)
R = float(arguments["--R"])
M_gas = float(arguments["--M"])
N_gas = int(float(arguments["--N"]) + 0.5)

tmax = int(float(arguments["--tmax"]))
nsnap = int(float(arguments["--nsnap"]))

magnetic_field = float(arguments["--bturb"])
bfixed = float(arguments["--bfixed"])

filename = arguments["--filename"]
diffuse_gas = not arguments["--no_diffuse_gas"]

param_only = arguments["--param_only"]

B_unit = float(arguments["--B_unit"])
length_unit = float(arguments["--length_unit"])
mass_unit = float(arguments["--mass_unit"])
v_unit = float(arguments["--v_unit"])
t_unit = length_unit / v_unit
G = 4300.71 * v_unit**-2 * mass_unit / length_unit
makebox = arguments["--makebox"]


if arguments["--glass_path"]:
    glass_path = arguments["--glass_path"]
else:
    glass_path = os.path.expanduser("~") + "/glass_orig.npy"
    if not os.path.exists(glass_path):
        import urllib.request

        print("Downloading glass file...")
        urllib.request.urlretrieve(
            "http://www.tapir.caltech.edu/~mgrudich/glass_orig.npy",
            glass_path,
            #            "https://data.obs.carnegiescience.edu/starforge/glass_orig.npy", glass_path
        )

if arguments["--boxsize"] is not None:
    boxsize = float(arguments["--boxsize"])
else:
    boxsize = 10 * R


derefinement = arguments["--derefinement"]

res_effective = int(N_gas ** (1.0 / 3.0) + 0.5)


delta_m = M_gas / N_gas
delta_m_solar = delta_m / mass_unit
rho_avg = 3 * M_gas / R**3 / (4 * np.pi)

# This doesn't really matter, but keeping it here just in case
if delta_m_solar < 0.1:  # if we're doing something marginally IMF-resolving
    softening = (
        3.11e-5  # ~6.5 AU, minimum sink radius is 2.8 times that (~18 AU)
    )
else:  # something more FIRE-like, where we rely on a sub-grid prescription turning gas into star particles
    softening = 0.1


tff = (3 * np.pi / (32 * G * rho_avg)) ** 0.5
L = (4 * np.pi * R**3 / 3) ** (1.0 / 3)  # volume-equivalent box size
vrms = arguments["--vel"]
#(6 / 5 * G * M_gas / R) ** 0.5 * turbulence**0.5


paramsfile = str(
    open(
        os.path.realpath(__file__).replace("MakeCloud.py", "params.txt"), "r"
    ).read()
)


replacements = {
    "NAME": filename.replace(".hdf5", ""),
    "DTSNAP": tff / nsnap,
    "MAXTIMESTEP": tff / (nsnap),
    "SOFTENING": softening,
    "GASSOFT": 2.0e-8,
    "TMAX": tff * tmax,
    "BOXSIZE": boxsize,
    "OUTFOLDER": "output",
    "BH_SEED_MASS": delta_m / 2.0,
    "UNIT_L": 3.085678e18 * length_unit,
    "UNIT_M": 1.989e33 * mass_unit,
    "UNIT_V": v_unit * 1e2,
    "UNIT_B": B_unit,
}

for k, r in replacements.items():
    paramsfile = paramsfile.replace(
        k, (r if isinstance(r, str) else "{:.2e}".format(r))
    )

open("params_" + filename.replace(".hdf5", "") + ".txt", "w").write(paramsfile)
if makebox:
    replacements_box = replacements.copy()
    replacements_box["NAME"] = filename.replace(".hdf5", "_BOX")
    replacements_box["BOXSIZE"] = L
    paramsfile = str(
        open(
            os.path.realpath(__file__).replace("MakeCloud.py", "params.txt"),
            "r",
        ).read()
    )
    for k in replacements_box.keys():
        paramsfile = paramsfile.replace(k, str(replacements_box[k]))
    open("params_" + filename.replace(".hdf5", "") + "_BOX.txt", "w").write(
        paramsfile
    )


if param_only:
    print("Parameters only run, exiting...")
    exit()



dm = M_gas / N_gas
mgas = np.repeat(dm, N_gas)

x = get_glass_coords(N_gas, glass_path)
Nx = len(x)
x = 2 * (x - 0.5)
print("Computing radii...")
r = cdist(x, [np.zeros(3)])[:, 0]
print("Done! Sorting coordinates...")
x = x[r.argsort()][:N_gas]
print("Done! Rescaling...")
x *= (float(Nx) / N_gas * 4 * np.pi / 3 / 8) ** (1.0 / 3) * R
print("Done! Recomupting radii...")
r = cdist(x, [np.zeros(3)])[:, 0]
x, r = x / r.max(), r / r.max()
print("Doing density profile...")
rnew = r ** (3.0 / (3 + density_exponent)) * R
x = x * (rnew / r)[:, None]
r = np.sum(x**2, axis=1) ** 0.5
r_order = r.argsort()
x, r = np.take(x, r_order, axis=0), r[r_order]


v = np.ones((N_gas, 3))* vrms  # start with constant velocity
print("Coordinates obtained!")

Mr = mgas.cumsum()
ugrav = G * np.sum(Mr / r * mgas)

B = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.ones(N_gas)]
vA_unit = (
    3.429e8
    * B_unit
    * (M_gas) ** -0.5
    * R**1.5
    * np.sqrt(4 * np.pi / 3)
    / v_unit
)  # alfven speed for unit magnetic field
uB = (
    0.5 * M_gas * vA_unit**2
)  # magnetic energy we would have for unit magnetic field
if bfixed > 0:
    B = B * bfixed
else:
    B = B * np.sqrt(
        magnetic_field * ugrav / uB
    )  # renormalize to desired magnetic energy
#print ("B field magnitude: %g" % (np.average(np.sum(B**2, axis=1)) ** 0.5))
#exit (0)  # exit here to avoid writing the file if we are just testing

#v = v - np.average(v, axis=0)
x = x - np.average(x, axis=0)

r, phi = np.sum(x**2, axis=1) ** 0.5, np.arctan2(x[:, 1], x[:, 0])
theta = np.arccos(x[:, 2] / r)
phi += phimode * np.sin(2 * phi) / 2
x = (
    r[:, np.newaxis]
    * np.c_[
        np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    ]
)



u = np.ones_like(mgas) * 0.101 / 2.0  # /2 needed because it is molecular

u = (
    np.ones_like(mgas) * (200 / v_unit) ** 2
)  # start with specific internal energy of (200m/s)^2, this is overwritten unless starting with restart flag 2###### #0.101/2.0 #/2 needed because it is molecular





if diffuse_gas:
    # assuming 10K vs 10^4K gas: factor of ~10^3 density contrast
    rho_warm = M_gas * 3 / (4 * np.pi * R**3) / 1000
    M_warm = (
        boxsize**3 - (4 * np.pi * R**3 / 3)
    ) * rho_warm  # mass of diffuse box-filling medium
    N_warm = int(M_warm / (M_gas / N_gas))
    if derefinement:
        x0 = get_glass_coords(N_gas, glass_path)
        Nx = len(x0)
        x0 = 2 * (x0 - 0.5)
        r0 = (x0 * x0).sum(1) ** 0.5
        x0, r0 = x0[r0.argsort()], r0[r0.argsort()]
        # first lay down the stuff within 3*R
        N_warm = int(
            4 * np.pi * rho_warm * (3 * R) ** 3 / 3 / dm
        )  # number of cells within 3R
        x_warm = (
            x0[:N_warm] * 3 * R / r0[N_warm - 1]
        )  # uniform density of cells within 3R
        x0 = x0[
            N_warm:
        ]  # now we take the ones outside the initial sphere and map them to a n(R) ~ R^-3 profile so that we get constant number of cells per log radius interval
        r0 = r0[N_warm:]
        rnew = 3 * R * np.exp(np.arange(len(x0)) / N_warm / 3)
        x_warm = np.concatenate([x_warm, (rnew / r0)[:, None] * x0], axis=0)
        x_warm = x_warm[np.max(np.abs(x_warm), axis=1) < boxsize / 2]
        N_warm = len(x_warm)
        R_warm = (x_warm * x_warm).sum(1) ** 0.5
        mgas = np.concatenate(
            [mgas, np.clip(dm * (R_warm / (3 * R)) ** 3, dm, np.inf)]
        )
    else:
        x_warm = boxsize * np.random.rand(N_warm, 3) - boxsize / 2
        if impact_dist == 0:
            x_warm = x_warm[np.sum(x_warm**2, axis=1) > R**2]
        N_warm = len(x_warm)
        mgas = np.concatenate(
            [mgas, np.repeat(mgas.sum() / len(mgas), N_warm)]
        )
    x = np.concatenate([x, x_warm])
    v = np.concatenate([v, np.zeros((N_warm, 3))])
    Bmag = np.average(np.sum(B**2, axis=1)) ** 0.5
    B = np.concatenate(
        [B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([0, 0, 1])]
    )
    u = np.concatenate([u, np.repeat(101.0, N_warm)])

    if makecylinder:
        # The magnetic field is paralell to the cylinder (true at low densities, so probably fine for IC)
        B_cyl = np.concatenate(
            [B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([1, 0, 0])]
        )
        # Add diffuse medium
        M_warm_cyl = (boxsize_cyl**3 - (4 * np.pi * R**3 / 3)) * rho_warm
        N_warm_cyl = int(M_warm_cyl / (M_gas / N_gas))
        x_warm = (
            boxsize_cyl * np.random.rand(N_warm_cyl, 3) - boxsize_cyl / 2
        )  # will be recentered later
        x_warm = x_warm[
            ~ind_in_cylinder(x_warm, L_cyl, R_cyl)
        ]  # keep only warm gas outside the cylinder
        # print("N_warm_cyl: %g N_warm_cyl_kept %g "%(N_warm_cyl,len(x_warm)))
        N_warm_cyl = len(x_warm)
        x_cyl = np.concatenate([x_cyl, x_warm])
        v_cyl = np.concatenate([v_cyl, np.zeros((N_warm, 3))])

else:
    N_warm = 0

rho = np.repeat(3 * M_gas / (4 * np.pi * R**3), len(mgas))
if diffuse_gas:
    rho[-N_warm:] /= 1000
h = (32 * mgas / rho) ** (1.0 / 3)

x += boxsize / 2  # cloud is always centered at (boxsize/2,boxsize/2,boxsize/2)
if makecylinder:
    x_cyl += boxsize_cyl / 2



print("Writing snapshot...")

F = h5py.File(filename, "w")
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [
    len(mgas),
    0,
    0,
    0,
    0,
    (1 if M_star > 0 else 0),
]
F["Header"].attrs["NumPart_Total"] = [
    len(mgas),
    0,
    0,
    0,
    0,
    (1 if M_star > 0 else 0),
]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas)
F["PartType0"].create_dataset("Coordinates", data=x)
F["PartType0"].create_dataset("Velocities", data=v)
F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(len(mgas)))
F["PartType0"].create_dataset("InternalEnergy", data=u)

if magnetic_field > 0.0:
    F["PartType0"].create_dataset("MagneticField", data=B)
F.close()

if makebox:
    F = h5py.File(filename.replace(".hdf5", "_BOX.hdf5"), "w")
    F.create_group("PartType0")
    F.create_group("Header")
    F["Header"].attrs["NumPart_ThisFile"] = [len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["NumPart_Total"] = [len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["MassTable"] = [M_gas / len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["BoxSize"] = (4 * np.pi * R**3 / 3) ** (1.0 / 3)
    F["Header"].attrs["Time"] = 0.0
    F["PartType0"].create_dataset("Masses", data=mgas[: len(mgas)])
    F["PartType0"].create_dataset(
        "Coordinates",
        data=np.random.rand(len(mgas), 3) * F["Header"].attrs["BoxSize"],
    )
    F["PartType0"].create_dataset("Velocities", data=np.zeros((len(mgas), 3)))
    F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(len(mgas)))
    F["PartType0"].create_dataset("InternalEnergy", data=u)
    if magnetic_field > 0.0:
        F["PartType0"].create_dataset("MagneticField", data=B[: len(mgas)])
    F.close()

