"""Pick one probe per (material, target n) from cases/probe_grid.csv and write the candidate case files.

usage (from the project root): python3 src/cases/write_cases.py
Rules:
  1. Candidates are probes with n <= 28500.
  2. For a target T, the 150 eV and the 300 eV probe closest in |log n - log T| are found.
  3. If both energies have a probe within 10 % of T, the 150 eV probe closest in log n is taken.
     Otherwise the probe closest in log n (any energy) is taken.
  4. relative_miss = (n - T) / T.
  5. Two targets that pick the same probe (same material, energy, radius) would give the same id. The one
     with the smaller |relative_miss| keeps the probe; the other is listed as not hit.
Each kept pick gets two case files: polar (theta 0..80 deg, 1 deg, phi 0) and stereographic
(theta 0..80 deg, 2 deg; phi 0..358 deg, 2 deg). Both use the [start, stop, num] form of run_case.py.
The writer refuses duplicate ids and refuses to overwrite an existing file.
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drivers"))
import paths  # noqa: E402

CASES = paths.CASES_DIR
CSV_PATH = CASES / "probe_grid.csv"
MANIFEST = CASES / "manifest.csv"
OLD_CASES = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD_CASES))
from materials import MATERIALS  # noqa: E402

TARGETS = (1000, 1500, 2200, 3300, 5000, 7500, 11000, 16000, 22000, 28000)
N_MAX = 28500
MATERIAL_HKL = {"silicon": (0, 0, 1), "copper": (1, 1, 1), "tungsten": (1, 1, 0)}
SCAN_TYPES = {
    "polar": dict(folder="polar", scan_name="polar_step_1deg",
                  scan=dict(type="theta", theta_deg=[0, 80, 81], phi_deg=0)),
    "stereographic": dict(folder="stereographic", scan_name="stereographic_step_2deg",
                          scan=dict(type="theta_phi", theta_deg=[0, 80, 41], phi_deg=[0, 358, 180])),
}
MANIFEST_FIELDS = ["id", "material", "orientation", "energy_eV", "radius_A", "atoms", "n", "target",
                   "relative_miss", "scan_type", "number_of_angles", "memory_16n2_GiB", "sha256"]


def read_probes():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        print("probe_grid.csv header:", reader.fieldnames)
        probes = [dict(material=r["material"], orientation=r["orientation"], energy=int(r["energy_eV"]),
                       radius=float(r["radius_A"]), atoms=int(r["atoms"]), n=int(r["n"]),
                       seconds=float(r["seconds"]))
                  for r in reader]
    return probes


def pick(material, target, probes):
    cands = [p for p in probes if p["material"] == material and p["n"] <= N_MAX]
    if not cands:
        return None
    dist = lambda p: abs(math.log(p["n"]) - math.log(target))
    near = [p for p in cands if abs(p["n"] - target) / target <= 0.10]
    if any(p["energy"] == 150 for p in near) and any(p["energy"] == 300 for p in near):
        return min((p for p in near if p["energy"] == 150), key=dist)
    return min(cands, key=dist)


def angles(scan):
    theta = len(np.linspace(*scan["theta_deg"]))
    phi = len(np.linspace(*scan["phi_deg"])) if isinstance(scan["phi_deg"], list) else 1
    return theta * phi


def case_id(material, orientation, energy, radius, scan_name):
    return (f"{material}_hkl_{orientation}_radius_{radius:g}A_energy_{energy}eV_depth_1_scan_{scan_name}")


def main():
    probes = read_probes()
    picks = []
    for material in MATERIAL_HKL:
        for target in TARGETS:
            p = pick(material, target, probes)
            if p is None:
                picks.append(dict(material=material, target=target, probe=None))
                continue
            picks.append(dict(material=material, target=target, probe=p,
                              relative_miss=(p["n"] - target) / target))
    # collisions: one probe for two targets
    by_probe = {}
    for k in picks:
        if k["probe"] is None:
            continue
        key = (k["material"], k["probe"]["energy"], k["probe"]["radius"])
        by_probe.setdefault(key, []).append(k)
    for key, group in by_probe.items():
        if len(group) > 1:
            group.sort(key=lambda k: (abs(k["relative_miss"]), k["target"]))
            keep = group[0]
            for k in group[1:]:
                k["dropped_same_probe_as"] = keep["target"]
    print("\npick table (all targets):")
    for k in picks:
        if k["probe"] is None:
            print(f"  {k['material']:9s} target {k['target']:6d}: no candidate n <= {N_MAX}")
        else:
            p = k["probe"]
            tag = f" NOT HIT (same probe as target {k['dropped_same_probe_as']})" if "dropped_same_probe_as" in k else ""
            print(f"  {k['material']:9s} target {k['target']:6d}: n={p['n']:6d} E={p['energy']} "
                  f"r={p['radius']:g} atoms={p['atoms']} miss={k['relative_miss']:+.4f}{tag}")
    kept = [k for k in picks if k["probe"] is not None and "dropped_same_probe_as" not in k]
    ids = {}
    rows = []
    for k in kept:
        p = k["probe"]
        orientation = "".join(str(i) for i in MATERIAL_HKL[k["material"]])
        for scan_type, spec in SCAN_TYPES.items():
            cid = case_id(k["material"], orientation, p["energy"], p["radius"], spec["scan_name"])
            if cid in ids:
                raise ValueError(f"duplicate case id {cid} for target {k['target']} and {ids[cid]}")
            ids[cid] = k["target"]
            case = dict(id=cid, material=k["material"], hkl=list(MATERIAL_HKL[k["material"]]),
                        radius_A=p["radius"], emitter_depth=1, energy_eV=p["energy"],
                        level=MATERIALS[k["material"]]["level"], scan=spec["scan"],
                        scan_name=spec["scan_name"], atoms=p["atoms"], matrix_order_probe=p["n"])
            path = CASES / spec["folder"] / f"{cid}.json"
            if path.exists():
                raise FileExistsError(f"{path} exists; refusing to overwrite")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(case, indent=1, sort_keys=True) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(dict(id=cid, material=k["material"], orientation=orientation, energy_eV=p["energy"],
                             radius_A=f"{p['radius']:g}", atoms=p["atoms"], n=p["n"], target=k["target"],
                             relative_miss=f"{k['relative_miss']:+.4f}", scan_type=scan_type,
                             number_of_angles=angles(spec["scan"]),
                             memory_16n2_GiB=f"{16 * p['n'] ** 2 / 2 ** 30:.3f}", sha256=digest))
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} manifest rows and {len(rows)} case files")


if __name__ == "__main__":
    main()
