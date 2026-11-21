"""Materials of the study: full-word names, crystal structure, lattice constants (A), emitter, core level.

structure keys follow ase.build.bulk. emitter: chemical symbol of the emitting atom.
level: core level written as in the first study ('3p', '4f', ...; no spin-orbit splitting).
Lattice constants: standard room-temperature values (same as the first study where the material is the same).
"""
MATERIALS = {
    'aluminium':          dict(formula='Al',   structure='fcc',       a=4.050,  emitter='Al', level='2p'),
    'nickel':             dict(formula='Ni',   structure='fcc',       a=3.524,  emitter='Ni', level='3p'),
    'copper':             dict(formula='Cu',   structure='fcc',       a=3.6149, emitter='Cu', level='3p'),
    'silver':             dict(formula='Ag',   structure='fcc',       a=4.086,  emitter='Ag', level='3d'),
    'gold':               dict(formula='Au',   structure='fcc',       a=4.078,  emitter='Au', level='4f'),
    'iron':               dict(formula='Fe',   structure='bcc',       a=2.866,  emitter='Fe', level='3p'),
    'molybdenum':         dict(formula='Mo',   structure='bcc',       a=3.147,  emitter='Mo', level='3d'),
    'tungsten':           dict(formula='W',    structure='bcc',       a=3.165,  emitter='W',  level='4f'),
    'titanium':           dict(formula='Ti',   structure='hcp',       a=2.950,  c=4.686, emitter='Ti', level='3p'),
    'cobalt':             dict(formula='Co',   structure='hcp',       a=2.507,  c=4.069, emitter='Co', level='3p'),
    'silicon':            dict(formula='Si',   structure='diamond',   a=5.431,  emitter='Si', level='2p'),
    'silicon_carbide':    dict(formula='SiC',  structure='zincblende', a=4.359, emitter='Si', level='2p'),
    'gallium_arsenide':   dict(formula='GaAs', structure='zincblende', a=5.653, emitter='Ga', level='3d'),
    'aluminium_nitride':  dict(formula='AlN',  structure='wurtzite',  a=3.110,  c=4.980, emitter='Al', level='2p'),
    'gallium_nitride':    dict(formula='GaN',  structure='wurtzite',  a=3.189,  c=5.185, emitter='Ga', level='3d'),
    'magnesium_oxide':    dict(formula='MgO',  structure='rocksalt',  a=4.210,  emitter='Mg', level='2p'),
    'nickel_oxide':       dict(formula='NiO',  structure='rocksalt',  a=4.177,  emitter='Ni', level='3p'),
    'copper_gold_alloy':  dict(formula='Cu3Au', structure='L12',     a=3.750,  emitter='Cu', level='2p'),
}
