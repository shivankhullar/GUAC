#!/usr/bin/env python
"""
clip_FIREcloud.py: Extract a spherical region centered on the peak gas density
                   from a FIRE/GIZMO HDF5 snapshot to create cloud ICs.

Usage: clip_FIREcloud.py [options]

Options:
    -h, --help                                                          Show this screen
    --snapdir=<snapdir>                                                 Snapshots in a snapdir subdirectory? [default: False]
    --path=<path>                                                       Path to snapshot directory [default: ./]
    --snap_num=<snap_num>                                               Snapshot number to clip [default: 100]
    --ic_path=<ic_path>                                                 Path to save the IC file [default: ./]
    --dist_cut_off=<dist_cut_off>                                       Extraction sphere radius in kpc [default: 0.1]
    --dist_cut_off_physical=<dist_cut_off_physical>                     Is dist_cut_off in physical kpc? (False = already in code/comoving units) [default: False]
    --ignore_stars=<ignore_stars>                                       Exclude PartType4 (stellar particles) [default: False]
    --ignore_dm=<ignore_dm>                                             Exclude PartType1 (dark matter) [default: False]
    --ignore_sinks=<ignore_sinks>                                       Exclude PartType5 (sink/BH particles) [default: False]
"""

from generic_utils.fire_utils import *
from cloud_utils.cloud_quants import *
from cloud_utils.cloud_utils import *
from cloud_utils.cloud_selection import *
from generic_utils.script_utils import *
from hybrid_sims_utils.read_snap import *
from docopt import docopt

import os


def clip_FIREcloud(snap_num, dist_cut_off, path, ic_path,
                   dist_cut_off_physical=False, snapdir=False,
                   ignore_stars=False, ignore_dm=False, ignore_sinks=False):
    """
    Extract a spherical region centered on the peak gas density from a snapshot.

    Parameters
    ----------
    snap_num : int
    dist_cut_off : float
        Sphere radius. Physical kpc if dist_cut_off_physical=True, else code/comoving units.
    path : str
        Directory containing the snapshot(s).
    ic_path : str
        Directory to write the output IC file.
    dist_cut_off_physical : bool
        Convert dist_cut_off from physical to code (comoving) units before applying.
    snapdir : bool
        If True, expect snapdir_XXX/snapshot_XXX.hdf5 layout.
    ignore_stars, ignore_dm, ignore_sinks : bool
        Skip the corresponding particle types entirely.
    """

    # ── Read input snapshot ────────────────────────────────────────────────
    if snapdir:
        file_name = os.path.join(path, f'snapdir_{snap_num:03d}', f'snapshot_{snap_num:03d}.hdf5')
    else:
        file_name = os.path.join(path, f'snapshot_{snap_num:03d}.hdf5')

    print(f'Reading {file_name} ...')
    f = h5py.File(file_name, 'r')

    header = {}
    gas    = {}
    dm1    = {}
    stars  = {}
    sinks  = {}

    for key in f.keys():
        if key == 'Header':
            for k in f[key].attrs.keys():
                header[k] = f[key].attrs[k]
        elif key == 'PartType0':
            for k in f[key].keys():
                gas[k] = np.array(f[key][k])
            print(f'  Gas:   {len(gas["Coordinates"])} particles')
        elif key == 'PartType1':
            if not ignore_dm:
                for k in f[key].keys():
                    dm1[k] = np.array(f[key][k])
                print(f'  DM:    {len(dm1["Coordinates"])} particles')
            else:
                print('  DM:    ignored')
        elif key == 'PartType4':
            if not ignore_stars:
                for k in f[key].keys():
                    stars[k] = np.array(f[key][k])
                print(f'  Stars: {len(stars["Coordinates"])} particles')
            else:
                print('  Stars: ignored')
        elif key == 'PartType5':
            if not ignore_sinks:
                for k in f[key].keys():
                    sinks[k] = np.array(f[key][k])
                print(f'  Sinks: {len(sinks["Coordinates"])} particles')
            else:
                print('  Sinks: ignored')

    f.close()

    # ── Center on maximum gas density ─────────────────────────────────────
    center = gas['Coordinates'][np.argmax(gas['Density'])]
    print(f'Center (peak gas density): {center}')

    # ── Convert cut-off radius to code (comoving) units ───────────────────
    if dist_cut_off_physical:
        cut_off_distance = convert_quant_from_physical(dist_cut_off, key="Coordinates",
                                                       a=header["Time"],
                                                       h=header["HubbleParam"])
        print(f'Cut-off: {dist_cut_off} kpc (physical) -> {cut_off_distance:.6g} (code units)')
    else:
        cut_off_distance = dist_cut_off
        print(f'Cut-off: {cut_off_distance} (code/comoving units)')

    # ── Select particles within the sphere ────────────────────────────────
    def sphere_inds(data):
        d = np.linalg.norm(data['Coordinates'] - center, axis=1)
        return np.where(d < cut_off_distance)[0]

    gas_inds  = sphere_inds(gas)
    dm1_inds  = sphere_inds(dm1)   if dm1   else np.array([], dtype=int)
    star_inds = sphere_inds(stars) if stars else np.array([], dtype=int)
    sink_inds = sphere_inds(sinks) if sinks else np.array([], dtype=int)

    print(f'Particles selected within sphere:')
    print(f'  Gas:   {len(gas_inds)}')
    if dm1:   print(f'  DM:    {len(dm1_inds)}')
    if stars: print(f'  Stars: {len(star_inds)}')
    if sinks: print(f'  Sinks: {len(sink_inds)}')

    # ── Build updated NumPart arrays ──────────────────────────────────────
    new_numpart = np.zeros(6, dtype=np.int32)
    new_numpart[0] = len(gas_inds)
    new_numpart[1] = len(dm1_inds)
    new_numpart[4] = len(star_inds)
    new_numpart[5] = len(sink_inds)

    # ── Write output IC file ───────────────────────────────────────────────
    os.makedirs(ic_path, exist_ok=True)
    out_name = os.path.join(ic_path, f'snapshot_{snap_num:03d}.hdf5')
    print(f'Writing IC to {out_name} ...')

    with h5py.File(out_name, 'w') as fout:
        hdr = fout.create_group('Header')
        for k, v in header.items():
            if k in ('NumPart_ThisFile', 'NumPart_Total'):
                hdr.attrs.create(k, new_numpart)
            elif k == 'NumPart_Total_HighWord':
                hdr.attrs.create(k, np.zeros(6, dtype=np.uint32))
            else:
                hdr.attrs.create(k, v)

        # Gas (always present)
        pt0 = fout.create_group('PartType0')
        for k in gas.keys():
            pt0.create_dataset(k, data=gas[k][gas_inds])

        # DM — only write group if particles survive the cut
        if len(dm1_inds) > 0:
            pt1 = fout.create_group('PartType1')
            for k in dm1.keys():
                pt1.create_dataset(k, data=dm1[k][dm1_inds])

        # Stars
        if len(star_inds) > 0:
            pt4 = fout.create_group('PartType4')
            for k in stars.keys():
                pt4.create_dataset(k, data=stars[k][star_inds])

        # Sinks
        if len(sink_inds) > 0:
            pt5 = fout.create_group('PartType5')
            for k in sinks.keys():
                pt5.create_dataset(k, data=sinks[k][sink_inds])

    print('Done.')


if __name__ == '__main__':
    args = docopt(__doc__)

    snapdir               = convert_to_bool(args['--snapdir'])
    path                  = args['--path']
    snap_num              = int(args['--snap_num'])
    ic_path               = args['--ic_path']
    dist_cut_off          = float(args['--dist_cut_off'])
    dist_cut_off_physical = convert_to_bool(args['--dist_cut_off_physical'])
    ignore_stars          = convert_to_bool(args['--ignore_stars'])
    ignore_dm             = convert_to_bool(args['--ignore_dm'])
    ignore_sinks          = convert_to_bool(args['--ignore_sinks'])

    print('Configuration:')
    print(f'  Path:           {path}')
    print(f'  Snap num:       {snap_num}')
    print(f'  IC path:        {ic_path}')
    print(f'  Cut-off radius: {dist_cut_off} ({"physical kpc" if dist_cut_off_physical else "code/comoving units"})')
    print(f'  Ignore DM:      {ignore_dm}')
    print(f'  Ignore stars:   {ignore_stars}')
    print(f'  Ignore sinks:   {ignore_sinks}')

    clip_FIREcloud(snap_num, dist_cut_off, path, ic_path,
                   dist_cut_off_physical=dist_cut_off_physical,
                   snapdir=snapdir,
                   ignore_stars=ignore_stars,
                   ignore_dm=ignore_dm,
                   ignore_sinks=ignore_sinks)
    print('IC file created successfully.')
