"""PHAGEN-only probe with stock MsSpec: matrix order n and atoms for a case (no PED solve).

usage: PYTHONPATH=<stock>/msspec_python3/src python3 src/cases/probe_n.py '<case json>'  -> one JSON line
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msspec.calculator import MSSPEC  # noqa: E402

from cluster import build_cluster  # noqa: E402
from materials import MATERIALS  # noqa: E402


def probe(case, workdir):
    """n at every energy of the case; matrix_order is the largest of them (the memory limit)."""
    cluster = build_cluster(case)
    for sub in ('input', 'output'):
        (Path(workdir) / sub).mkdir(parents=True, exist_ok=True)
    calc = MSSPEC(spectroscopy='PED', algorithm='inversion', folder=str(workdir), txt=None)
    calc.set_atoms(cluster)
    calc.scan_parameters.set_parameter('type', 'theta_phi')
    calc.spectroscopy_parameters.set_parameter('level', case.get('level') or MATERIALS[case['material']]['level'])
    orders = []
    t0 = time.perf_counter()
    try:
        for energy in case.get('energies_eV', [case['energy_eV']]):
            calc.scan_parameters.set_parameter('kinetic_energy', energy)
            calc.get_tmatrix()
            orders.append(int(max(sum(len(calc.phagenio.tl[int(a.get('proto_index')) - 1][ie]) ** 2 for a in calc.atoms)
                                  for ie in range(calc.phagenio.ne))))
    finally:
        calc.shutdown()
    return {'atoms': len(cluster), 'matrix_order': max(orders), 'matrix_order_per_energy': orders,
            'phagen_seconds': time.perf_counter() - t0}


if __name__ == '__main__':
    case = json.loads(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix='probe-') as d:
        print(json.dumps(dict(case, **probe(case, d))))
