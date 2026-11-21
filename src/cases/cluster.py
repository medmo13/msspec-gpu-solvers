"""Build a PED cluster: half-space below a low-index surface, sphere of radius R around the emitter.

case keys: material, hkl (Miller indices of the surface; for hexagonal crystals the 3-index form
(h, k, l) of the reciprocal basis), radius_A, emitter_depth (1 = top layer that holds the emitter species),
orientation_deg (optional Euler angles x, y, z applied last; used for random orientations).
Uses only ase and the stock msspec helpers (center_cluster, get_atom_index).
"""
import math

import numpy as np
from ase import Atoms
from ase.build import bulk

from materials import MATERIALS


def bulk_cell(name):
    m = MATERIALS[name]
    if m['structure'] == 'L12':
        a = m['a']
        solid = Atoms('Cu3Au', scaled_positions=[[.5, .5, 0], [.5, 0, .5], [0, .5, .5], [0, 0, 0]],
                      cell=[a] * 3, pbc=True)
        return solid
    kw = dict(crystalstructure=m['structure'], a=m['a'])
    if 'c' in m:
        kw['c'] = m['c']
    if m['structure'] in ('fcc', 'bcc', 'diamond', 'zincblende', 'rocksalt'):
        kw['cubic'] = True
    return bulk(m['formula'], **kw)


def build_cluster(case):
    from msspec.utils import get_atom_index  # noqa: F401  (kept for callers that need it)
    name, radius = case['material'], float(case['radius_A'])
    m = MATERIALS[name]
    cell = bulk_cell(name)
    cell.set_tags([0 if s == m['emitter'] else 1 for s in cell.get_chemical_symbols()])
    # surface normal from the reciprocal lattice (cubic: [hkl]; hexagonal: (h k l) in the reciprocal basis)
    normal = np.asarray(case['hkl'], float) @ cell.cell.reciprocal()
    normal /= np.linalg.norm(normal)
    heights = [abs(np.dot(cell.cell[i], np.cross(cell.cell[(i + 1) % 3], cell.cell[(i + 2) % 3])))
               / np.linalg.norm(np.cross(cell.cell[(i + 1) % 3], cell.cell[(i + 2) % 3])) for i in range(3)]
    reach = radius + 3 * max(cell.cell.lengths()) + 3.0
    reps = int(math.ceil(2 * reach / min(heights))) + 1
    block = cell.repeat((reps, reps, reps))
    block.positions -= block.positions.mean(axis=0)
    block.rotate(normal, (0, 0, 1), center=(0, 0, 0))
    z = block.positions[:, 2]
    tags = block.get_tags()
    # planes of the lattice near z = 0 (use a lateral window so that corners of the block do not matter)
    window = np.hypot(block.positions[:, 0], block.positions[:, 1]) < reach * 0.5
    levels = np.unique(np.round(z[window & (np.abs(z) < 2 * max(cell.cell.lengths()))], 2))
    levels = levels[::-1]
    surface = levels[np.argmin(np.abs(levels))]            # top plane = the plane nearest the middle of the block
    below = [lv for lv in levels if lv <= surface + 1e-6]
    holding = [lv for lv in below if np.any((np.abs(z - lv) < 0.02) & (tags == 0) & window)]
    emitter_level = holding[int(case.get('emitter_depth', 1)) - 1]
    candidates = np.where((np.abs(z - emitter_level) < 0.02) & (tags == 0))[0]
    lateral = np.hypot(block.positions[candidates, 0], block.positions[candidates, 1])
    emitter = candidates[int(np.argmin(lateral))]
    block.positions -= block.positions[emitter]
    surface_rel = surface - emitter_level
    r = np.linalg.norm(block.positions, axis=1)
    keep = (r <= radius) & (block.positions[:, 2] <= surface_rel + 0.02)
    cluster = block[keep]
    cluster.absorber = int(np.where(np.linalg.norm(cluster.positions, axis=1) < 1e-6)[0][0])
    for angle, axis in zip(case.get('orientation_deg', (0, 0, 0)), ('x', 'y', 'z')):
        if angle:
            cluster.rotate(angle, axis, rotate_cell=False)
    cluster.set_pbc(False)
    return cluster
