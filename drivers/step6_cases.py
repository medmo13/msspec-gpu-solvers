"""Step 6 case step: rebuild check of the existing 58 case files with the generator's functions, live probe of the
silicon candidates in the n window (20875, 28383), then write the missing silicon 10th-n case pair (polar and
stereographic) and append its two manifest rows (append only; write_cases.main() is NOT called: it rewrites the
whole manifest).
Generator reused by import: src/cases/write_cases.py (case_id, SCAN_TYPES, MANIFEST_FIELDS, angles, MATERIALS)
and src/cases/probe_grid.py (run_probe, the generator's own probe; probe_grid.csv is not written).
usage: TMPDIR=$REPO/.work python3 -B drivers/step6_cases.py
"""
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import paths  # noqa: E402

V2 = paths.REPO
sys.path.insert(0, str(V2 / "src" / "cases"))
import write_cases as wc  # noqa: E402
import probe_grid as pg  # noqa: E402

TMP = paths.WORK_DIR / "step6_probe"
TMP.mkdir(parents=True, exist_ok=True)
pg.TMP = TMP
os.environ["TMPDIR"] = str(TMP)
WINDOW = (20875, 28383)


def build_case(material, energy, radius, atoms, n, scan_type):
    """Same dict as write_cases.main() builds for a kept pick."""
    spec = wc.SCAN_TYPES[scan_type]
    cid = wc.case_id(material, "".join(str(i) for i in wc.MATERIAL_HKL[material]), energy, radius, spec["scan_name"])
    case = dict(id=cid, material=material, hkl=list(wc.MATERIAL_HKL[material]), radius_A=radius, emitter_depth=1,
                energy_eV=energy, level=wc.MATERIALS[material]["level"], scan=spec["scan"],
                scan_name=spec["scan_name"], atoms=atoms, matrix_order_probe=n)
    return cid, json.dumps(case, indent=1, sort_keys=True) + "\n"


def step_rebuild_check():
    rows = list(csv.DictReader(wc.MANIFEST.open(newline="")))
    bad = 0
    for r in rows:
        cid, text = build_case(r["material"], int(r["energy_eV"]), float(r["radius_A"]), int(r["atoms"]),
                               int(r["n"]), r["scan_type"])
        path = wc.CASES / wc.SCAN_TYPES[r["scan_type"]]["folder"] / f"{cid}.json"
        same_id = cid == r["id"]
        same_bytes = path.read_text() == text
        same_sha = hashlib.sha256(path.read_bytes()).hexdigest() == r["sha256"]
        if not (same_id and same_bytes and same_sha):
            bad += 1
            print("REBUILD MISMATCH", r["id"], same_id, same_bytes, same_sha)
    print(f"rebuild check: {len(rows)} manifest rows, mismatches {bad}")
    if bad:
        raise SystemExit("rebuild check failed")


def main():
    assert os.environ.get("TMPDIR") == str(TMP), os.environ.get("TMPDIR")
    print("TMP", TMP)
    step_rebuild_check()
    # live probes of the two silicon candidates from the grid inside the window
    cands = []
    for energy, radius in [(300, 10.5), (300, 11.0)]:
        result, seconds = pg.run_probe("silicon", (0, 0, 1), energy, radius)
        print(f"live probe silicon 001 E={energy} r={radius}: atoms={result['atoms']} n={result['matrix_order']} "
              f"seconds={seconds:.3f}")
        cands.append(dict(energy=energy, radius=radius, atoms=result["atoms"], n=result["matrix_order"]))
    inside = [c for c in cands if WINDOW[0] < c["n"] < WINDOW[1]]
    mid = math.sqrt(WINDOW[0] * WINDOW[1])
    pick = min(inside, key=lambda c: abs(math.log(c["n"]) - math.log(mid)))
    print(f"window {WINDOW}, log-midpoint {mid:.1f}, pick by closest log n: {pick}")
    cid_p, text_p = build_case("silicon", pick["energy"], pick["radius"], pick["atoms"], pick["n"], "polar")
    cid_s, text_s = build_case("silicon", pick["energy"], pick["radius"], pick["atoms"], pick["n"], "stereographic")
    print("new ids:", cid_p, cid_s)
    paths = [wc.CASES / "polar" / f"{cid_p}.json", wc.CASES / "stereographic" / f"{cid_s}.json"]
    for p in paths:
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    existing = list(csv.DictReader(wc.MANIFEST.open(newline="")))
    if any(r["id"] in (cid_p, cid_s) for r in existing):
        raise SystemExit("id already in manifest")
    new_rows = []
    for scan_type, path, cid, text in [("polar", paths[0], cid_p, text_p),
                                       ("stereographic", paths[1], cid_s, text_s)]:
        path.write_text(text)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = wc.SCAN_TYPES[scan_type]
        new_rows.append(dict(id=cid, material="silicon", orientation="001", energy_eV=pick["energy"],
                             radius_A=f"{pick['radius']:g}", atoms=pick["atoms"], n=pick["n"], target="",
                             relative_miss="", scan_type=scan_type,
                             number_of_angles=wc.angles(spec["scan"]),
                             memory_16n2_GiB=f"{16 * pick['n'] ** 2 / 2 ** 30:.3f}", sha256=digest))
    with wc.MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wc.MANIFEST_FIELDS)
        w.writerows(new_rows)
    print("appended", len(new_rows), "manifest rows; files written:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
