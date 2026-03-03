"""
Some functions taken from Mike Grudic's starforge_tools repository and modified slightly,

Author: Shivan Khullar
Date: March 2026
"""

import h5py
import numpy as np




def extract_stellar_masses_from_sim(sim_dict):
    """
    Extract stellar masses from all snapshots in a simulation.
    Returns the maximum mass each star ever reached (ZAMS mass).
    """
    zams_mass_dict = {}
    t0_dict = {}
    
    sim_path = sim_dict['sim_full_path']
    snaps = glob.glob(os.path.join(sim_path, "snapshot_*.hdf5"))
    
    if len(snaps) == 0:
        print(f"No snapshots found for {sim_dict['label']}")
        return np.array([])
    
    for s in sorted(snaps):
        try:
            with h5py.File(s, "r") as F:
                if "PartType5" not in F.keys():
                    continue
                ids = F["PartType5/ParticleIDs"][:]
                mstar = F["PartType5/BH_Mass"][:]
                t = F["Header"].attrs["Time"]
                
                for i in range(len(ids)):
                    star_id, star_mass = ids[i], mstar[i]
                    if star_id not in zams_mass_dict.keys():
                        zams_mass_dict[star_id] = star_mass
                        t0_dict[star_id] = t
                    else:
                        zams_mass_dict[star_id] = max(star_mass, zams_mass_dict[star_id])
                        t0_dict[star_id] = min(t, t0_dict[star_id])
        except Exception as e:
            print(f"Could not open {s}: {e}")
            continue
    
    masses = np.array([zams_mass_dict[i] for i in zams_mass_dict.keys()])
    print(f"{sim_dict['label']}: Found {len(masses)} stars, total mass = {masses.sum():.2f} Msun")
    
    return masses




def compute_imf_histogram(masses, bin_edges=None, normalize=True):
    """
    Compute IMF histogram (dN/dlog(m) vs m).
    
    Parameters:
    -----------
    masses : array
        Stellar masses
    bin_edges : array, optional
        Mass bin edges. If None, uses logarithmic bins
    normalize : bool
        If True, normalize by bin width to get dN/dlog(m)
    
    Returns:
    --------
    bin_centers : array
        Center of mass bins
    imf_values : array
        IMF values (dN/dlog(m) if normalized)
    bin_edges : array
        Edges of bins used
    """
    if len(masses) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if bin_edges is None:
        # Create logarithmic bins from min to max mass
        nbins = 20
        #max(10, min(int(np.sqrt(len(masses))), 50))
        #bin_edges = np.logspace(np.log10(0.01), np.log10(100), nbins) #np.logspace(np.log10(masses.min()), np.log10(masses.max()), nbins)
        bin_edges = np.logspace(np.log10(0), np.log10(100), nbins) #np.logspace(np.log10(masses.min()), np.log10(masses.max()), nbins)
    
    counts, _ = np.histogram(masses, bins=bin_edges)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
    
    if normalize:
        # dN/dlog(m)
        dlogm = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        imf_values = counts / dlogm
    else:
        imf_values = counts
    
    # Remove empty bins
    valid = imf_values > 0
    
    return bin_centers[valid], imf_values[valid], bin_edges


def get_imf_hist(masses, bins=None, mode='dn_dm'):
    if bins is None:
        #bins = np.logspace(-2, 2, 20)
        bins = np.logspace(-2, 3, 20)
    
    #n, b = np.histogram(masses, bins=bins, density=True)
    n, b = np.histogram(masses, bins=bins, density=False)
    
    dm = np.diff(b)
    dlogm = np.diff(np.log10(b))

    dn_dlogm = n / dlogm  # dN/d(log m)
    dn_dm = n / dm      # dN/dm


    bin_centers = np.zeros(len(b)-1)
    for i in range(0, len(b)-1):
        bin_centers[i] = (b[i] + b[i+1]) / 2

    if mode=='dn_dlogm':
        return bin_centers, dn_dlogm
    elif mode=='dn_dm':
        return bin_centers, dn_dm
    else:
        raise ValueError("Invalid mode")
    



def simple_powerlaw_fit(
    data, xmin=None, xmax=None, Ngrid=1024, alpha_min=-4, alpha_max=4, quantiles=[0.16, 0.5, 0.84], return_grid=False
):
    """Computes the specified quantiles of the posterior distribution of the power law slope alpha, given a dataset and desired quantiles

    Example usage to get quantiles on the power-law slope of the IMF over the interval [1,10]:
    `
    masses = np.loadtxt("my_IMF_data.dat")
    quantiles = simple_powerlaw_fit(masses, xmin=1, xmax=10)
    `
    """
    if xmin is None:
        xmin = data.min()
    if xmax is None:
        xmax = data.max()
    data = data[(data > xmin) * (data < xmax)]  # prune data to specified limits

    alpha_grid = np.linspace(alpha_min, alpha_max, Ngrid)  # initialize the 1D grid in parameter space
    lnprob = np.zeros_like(alpha_grid)  # grid that stores the values of the posterior distribution

    normgrid = (1 + alpha_grid) / (
        xmax ** (1 + alpha_grid) - xmin ** (1 + alpha_grid)
    )  # grid of normalization of the distribution x^alpha over the limits [mmin,mmax]

    for d in data:  # sum the posterior log-likelihood distribution
        lnprob += np.log(d**alpha_grid * normgrid)

    # convert log likelihood to likelihood
    lnprob -= lnprob.max()
    prob = np.exp(lnprob)  # watch for overflow errors here
    prob /= np.trapezoid(prob, alpha_grid)  # normalize

    q = np.interp(quantiles, np.cumsum(prob) / prob.sum(), alpha_grid)
    if return_grid:
        return q, alpha_grid, prob
    else:
        return q  # returns quantiles of the posterior distribution
    