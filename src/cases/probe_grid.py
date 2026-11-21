"""PHAGEN-only probe grid for the V2 candidate case set: matrix order n over cluster radius.

For each material (fixed low-index surface) and each energy (150 eV, 300 eV), the cluster radius goes from
3.0 A to 12.0 A in 0.5 A steps, emitter depth 1. A series stops after the first probe with n > 30000.
Every probe is appended to cases/probe_grid.csv and flushed at once.
Resumable: (material, energy, radius) rows already in the CSV are skipped.
No PED solve is run. The probe is the stock-MsSpec probe of old src/cases/probe_n.py (imported, not edited).

usage (from the project root):
  python3 src/cases/probe_grid.py > .work/tmp_cases/probe_grid.log 2>&1
"""
import os

THREADS = "4"  # modest CPU use: other workers are building at the same time
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, THREADS)

import csv  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drivers"))
import paths  # noqa: E402

STOCK_SRC = paths.stock_pythonpath()
OLD_CASES = Path(__file__).resolve().parent
TMP = paths.WORK_DIR / "tmp_cases"
CSV_PATH = paths.CASES_DIR / "probe_grid.csv"
os.environ["TMPDIR"] = str(TMP)

sys.path.insert(0, str(STOCK_SRC))
sys.path.insert(0, str(OLD_CASES))
from materials import MATERIALS  # noqa: E402
from probe_n import probe  # noqa: E402

MATERIAL_HKL = {"silicon": (0, 0, 1), "copper": (1, 1, 1), "tungsten": (1, 1, 0)}
ENERGIES = (150, 300)
RADII = tuple(round(3.0 + 0.5 * i, 1) for i in range(19))  # 3.0 .. 12.0 A
N_STOP = 30000
FIELDS = ["material", "orientation", "energy_eV", "radius_A", "emitter_depth", "atoms", "n", "seconds",
          "phagen_seconds"]


def load_rows():
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def run_probe(material, hkl, energy, radius):
    case = dict(material=material, hkl=list(hkl), radius_A=radius, emitter_depth=1, energy_eV=energy,
                level=MATERIALS[material]["level"])
    t0 = time.perf_counter()                       # end to end: workdir, cluster, calculator, PHAGEN, shutdown
    workdir = Path(tempfile.mkdtemp(prefix="probe_", dir=TMP))
    try:
        result = probe(case, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    seconds = time.perf_counter() - t0
    return result, seconds


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("temp dir:", TMP, flush=True)
    print("csv:", CSV_PATH, flush=True)
    rows = load_rows()
    done = {(r["material"], int(r["energy_eV"]), float(r["radius_A"])) for r in rows}
    fresh = not CSV_PATH.exists()
    start = time.perf_counter()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
            f.flush()
        for material, hkl in MATERIAL_HKL.items():
            for energy in ENERGIES:
                series = [r for r in rows if r["material"] == material and int(r["energy_eV"]) == energy]
                if any(int(r["n"]) > N_STOP for r in series):
                    print(f"series {material} {energy} eV already finished (n > {N_STOP})", flush=True)
                    continue
                for radius in RADII:
                    if (material, energy, radius) in done:
                        continue
                    result, seconds = run_probe(material, hkl, energy, radius)
                    row = dict(material=material, orientation="".join(str(i) for i in hkl), energy_eV=energy,
                               radius_A=f"{radius:.1f}", emitter_depth=1, atoms=result["atoms"],
                               n=result["matrix_order"], seconds=f"{seconds:.3f}",
                               phagen_seconds=f"{result['phagen_seconds']:.3f}")
                    writer.writerow(row)
                    f.flush()
                    print(f"probe {material} hkl={row['orientation']} E={energy} r={row['radius_A']} "
                          f"atoms={row['atoms']} n={row['n']} seconds={row['seconds']}", flush=True)
                    if result["matrix_order"] > N_STOP:
                        print(f"series {material} {energy} eV stops: n > {N_STOP}", flush=True)
                        break
    print(f"grid loop finished in {time.perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
