#!/usr/bin/env python3
"""
min_gas_resolution.py: Find the minimum gas spatial resolution (dm/rho)^(1/3) and the maximum
gas density (g/cm^3 and particles/cm^3, n = rho/m_p) across snapshots.

Run from the snapshot directory or pass --snapdir explicitly.

Usage: min_gas_resolution.py [options]

Options:
    -h, --help                      Show this screen
    --snapdir=<snapdir>             Directory containing snapshots [default: .]
"""

import sys
import os
import glob
import re
import numpy as np
from docopt import docopt

from hybrid_sims_utils.read_snap import get_snap_data_hybrid, convert_units_to_physical
from generic_utils.constants import Msun, kpc, m_p

# Physical code density unit (1e10 Msun / kpc^3) in g/cm^3
UNIT_DENSITY_CGS = 1e10 * Msun / kpc**3


def get_snap_numbers(snapdir):
    pattern = os.path.join(snapdir, 'snapshot_*.hdf5')
    files = sorted(glob.glob(pattern))
    snap_nums = []
    for f in files:
        m = re.search(r'snapshot_(\d+)\.hdf5$', f)
        if m:
            snap_nums.append(int(m.group(1)))
    return sorted(snap_nums)


def min_resolution_for_snap(snapdir, snap_num):
    header, pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_hybrid(
        sim=snapdir,
        sim_path='',
        snap=snap_num,
        snapdir=False,
        custom_gas_fields=['Masses', 'Density'],
        ignore_data_type=['sinks', 'stars', 'refine'],
        verbose=False,
    )

    if not pdata or 'Masses' not in pdata or 'Density' not in pdata:
        print(f'  Snapshot {snap_num}: no gas data found, skipping')
        return None

    header, pdata, _, _ = convert_units_to_physical(header, pdata, {}, {})

    # (mass / density)^(1/3) in kpc; convert to pc and AU (1 pc = 206265 AU)
    res_kpc = (pdata['Masses'] / pdata['Density']) ** (1.0 / 3.0)
    res_pc = res_kpc * 1e3
    res_au = res_pc * 206265.0

    # Maximum density in g/cm^3, and as a number density n = rho / m_p in cm^-3
    rho_max_cgs = float(pdata['Density'].max()) * UNIT_DENSITY_CGS
    return float(res_pc.min()), float(res_au.min()), rho_max_cgs, rho_max_cgs / m_p


if __name__ == '__main__':
    args = docopt(__doc__)
    snapdir = args['--snapdir']

    snap_nums = get_snap_numbers(snapdir)
    if not snap_nums:
        print(f'No snapshots found in {snapdir}')
        sys.exit(1)

    print(f'Found {len(snap_nums)} snapshots in {snapdir}')
    print(f'Snapshot range: {snap_nums[0]} - {snap_nums[-1]}')
    print()

    global_min_pc = np.inf
    global_min_au = np.inf
    global_min_snap = -1
    global_max_rho = -np.inf
    global_max_n = -np.inf
    global_max_snap = -1

    for snap_num in snap_nums:
        result = min_resolution_for_snap(snapdir, snap_num)
        if result is None:
            continue
        min_pc, min_au, max_rho, max_n = result
        print(f'Snapshot {snap_num:04d}:  min resolution = {min_pc:.3e} pc  ({min_au:.2f} AU)  |  '
              f'max density = {max_rho:.3e} g/cm^3  ({max_n:.3e} cm^-3)')
        if min_pc < global_min_pc:
            global_min_pc = min_pc
            global_min_au = min_au
            global_min_snap = snap_num
        if max_rho > global_max_rho:
            global_max_rho = max_rho
            global_max_n = max_n
            global_max_snap = snap_num

    print()
    print(f'Global minimum resolution: {global_min_pc:.3e} pc  ({global_min_au:.2f} AU)  (snapshot {global_min_snap:04d})')
    print(f'Global maximum density:    {global_max_rho:.3e} g/cm^3  ({global_max_n:.3e} cm^-3)  (snapshot {global_max_snap:04d})')
