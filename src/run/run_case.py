"""Run one study case (stock MsSpec API, algorithm='inversion') and save the cross section.

usage: PYTHONPATH=<version>/msspec_python3/src MPLBACKEND=Agg python3 src/run/run_case.py <case.json> <output_dir>
Writes <output_dir>/result.npz (theta, phi, cross_section; one block per energy) and <output_dir>/meta.json
(matrix order n from PHAGEN, atoms, energies, in-process seconds). The whole-process time is measured by the caller
around this process, not here. The solver is chosen by MSSPEC_SOLVER (unset = stock LAPACK).
The script refuses to overwrite an existing result.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from msspec.calculator import MSSPEC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cases"))
from cluster import build_cluster  # noqa: E402


def matrix_order(calc):
    return int(max(sum(len(calc.phagenio.tl[int(a.get("proto_index")) - 1][ie]) ** 2 for a in calc.atoms)
                   for ie in range(calc.phagenio.ne)))


def main():
    case = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.npz").exists():
        raise FileExistsError(f"{out / 'result.npz'} exists; choose a new output directory")
    t_start = time.perf_counter()
    cluster = build_cluster(case)
    calc = MSSPEC(spectroscopy="PED", algorithm="inversion", folder=str(out / "calc"), txt=None)
    calc.set_atoms(cluster)
    scan, blocks, orders = case["scan"], {}, []
    energies = case.get("energies_eV", [case["energy_eV"]])
    try:
        for k, energy in enumerate(energies):
            theta = np.linspace(*scan["theta_deg"])
            if scan["type"] == "theta_phi":
                data = calc.get_theta_phi_scan(level=case["level"], kinetic_energy=energy, theta=theta,
                                               phi=np.linspace(*scan["phi_deg"]))
            else:
                data = calc.get_theta_scan(level=case["level"], kinetic_energy=energy, theta=theta,
                                           phi=scan["phi_deg"])
            last = data.get_last_dset()
            for key in ("theta", "phi", "cross_section"):
                blocks[f"{key}_{k}"] = np.asarray(last[key])
            orders.append(matrix_order(calc))
    finally:
        calc.shutdown()
    if not all(np.isfinite(v).all() for v in blocks.values()):
        raise RuntimeError("non-finite result; no points are removed")
    np.savez_compressed(out / "result.npz", **blocks)
    (out / "meta.json").write_text(json.dumps(dict(
        case=case["id"], atoms=len(cluster), energies_eV=energies, matrix_order=orders,
        matrix_order_probe=case.get("matrix_order_probe"), in_process_seconds=time.perf_counter() - t_start,
    ), indent=1) + "\n")
    print("done", case["id"], "n", orders, "points", sum(v.size for k, v in blocks.items() if k.startswith("cross")))


if __name__ == "__main__":
    main()
