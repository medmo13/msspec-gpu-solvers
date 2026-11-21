"""Step 6: polar (30 cases) and stereographic (30 cases) study, 3 materials x 10 n, one calculation per method per case.

usage: TMPDIR=$REPO/.work python3 -B drivers/step6_run.py
Phases (sequential, small n first inside each phase): polar, then stereographic.
Reuse, not edit: run_one of drivers/step5_run.py (in-memory CASE_DIR set to the phase folder), rg.preflight,
ts.wait_for_idle, s3.child_env (identical to rg.child_env, checked), src/check_accuracy.py.
Methods: polar = stock + 8 GPU variants; stereographic = stock + own_refined + own_unrefined.
Per case: one untimed wake-up (own_unrefined), then one timed run of each method in the order
random.Random(20261118).shuffle(list) (fresh generator per case, base list as in METHODS below).
Stop limits (step 5 rule): stock none; GPU runs max(2 x stock s, 60 s) when stock already ran in this case, else 3600 s.
Reused rows (not rerun): polar copper n 1210, 3411, 7774, 15851, 26741 (rg.CASES); stereographic copper (s5.CASES).
Outputs:
  results/step6/polar/<case>/<method>/ and results/step6/stereo/<case>/<method>/ (result.npz, stdout/stderr)
  results/step6/polar_runs.jsonl, stereo_runs.jsonl        (one line per run, raw, from step5_run.run_one)
  results/step6/polar_accuracy.jsonl, stereo_accuracy.jsonl (per timed run: max|d| vs this case's stock, pass)
  results/step6/all_polar.jsonl                             (reused + new timed rows, field 'source')
  results/step6/polar_done.txt, stereo_done.txt             (FINISHED or ABORTED: ...)
  results/step6/driver.log                                  (progress, written by the shell redirect)
"""
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
import paths  # noqa: E402
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(HERE))
import time_stock as ts  # noqa: E402
import run_gpu_variants as rg  # noqa: E402
import step3b_run as s3  # noqa: E402
import step5_run as s5  # noqa: E402

OUT = paths.RESULTS_DIR / "step6"
MANIFEST = paths.CASES_DIR / "manifest.csv"
SEED = rg.SEED
TOL = rg.TOL
WAKE = "own_unrefined"
POLAR_METHODS = ["stock", "cusolver_refined", "cusolver_unrefined", "magma_refined", "magma_unrefined",
                 "magma_host_refined", "magma_host_unrefined", "own_refined", "own_unrefined"]
STEREO_METHODS = ["stock", "own_refined", "own_unrefined"]
REUSED_POLAR = list(rg.CASES)
REUSED_STEREO = list(s5.CASES)


def manifest_rows(scan_type):
    with MANIFEST.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["scan_type"] == scan_type]
    if len(rows) != 30:
        raise SystemExit(f"{scan_type}: {len(rows)} manifest rows, expected 30")
    return sorted(rows, key=lambda r: int(r["n"]))


def new_cases(scan_type, reused):
    rows = manifest_rows(scan_type)
    ids = {r["id"] for r in rows}
    missing = [c for c in reused if c not in ids]
    if missing:
        raise SystemExit(f"reused cases not in manifest: {missing}")
    return [r for r in rows if r["id"] not in reused]


def case_json(folder, case_id):
    return json.loads((paths.CASES_DIR / folder / f"{case_id}.json").read_text())


def accuracy_for_case(recs, material, scan_type, case_id):
    timed = [r for r in recs if r["kind"] == "timed"]
    stock = [r for r in timed if r["method"] == "stock"]
    st = stock[0] if stock and stock[0]["status"] == "ok" else None
    stock_s = st["seconds"] if st else None
    out = []
    for r in timed:
        rec = dict(case=case_id, material=material, scan_type=scan_type, method=r["method"], status=r["status"],
                   rc=r["rc"], seconds=r["seconds"], n=r["n"], n_equals_probe=r.get("n_equals_probe"),
                   stock_seconds=stock_s, speedup=None, max_d=None, **{"pass": None}, accuracy_error=None)
        if r["status"] == "ok" and stock_s is not None:
            rec["speedup"] = stock_s / r["seconds"]
        if r["method"] != "stock" and r["status"] == "ok" and st is not None:
            test = Path(r["run_dir"]) / "result.npz"
            ref = Path(st["run_dir"]) / "result.npz"
            ac = subprocess.run([sys.executable, "-B", str(V2 / "src" / "check_accuracy.py"), str(test), str(ref)],
                                capture_output=True, text=True)
            (Path(r["run_dir"]) / "accuracy.txt").write_text(ac.stdout + ac.stderr)
            m = re.search(r"max\|d\| over all blocks = (\S+?),", ac.stdout)
            if m is None:
                rec["accuracy_error"] = f"no max|d| line, check_accuracy rc {ac.returncode}"
            else:
                rec["max_d"] = float(m.group(1))
                rec["pass"] = rec["max_d"] <= TOL
        out.append(rec)
    return out


def run_phase(name, folder, scan_type, methods, reused, log):
    phase_dir = OUT / name
    s5.CASE_DIR = paths.CASES_DIR / folder           # in-memory config of the reused step 5 run_one
    runs = (OUT / f"{name}_runs.jsonl").open("x")
    acc = (OUT / f"{name}_accuracy.jsonl").open("x")
    all_recs = {}
    try:
        for row in new_cases(scan_type, reused):
            case_id, n_probe, material = row["id"], int(row["n"]), row["material"]
            cj = case_json(folder, case_id)
            assert int(cj["matrix_order_probe"]) == n_probe
            order = methods[:]
            random.Random(SEED).shuffle(order)
            rg.preflight()
            mem = s3.meminfo("MemAvailable")
            swap = s3.meminfo("SwapFree")
            print(f"[{name}] case {case_id} material {material} n {n_probe} order {order} "
                  f"MemAvailable {mem} SwapFree {swap}", file=log, flush=True)
            recs = []

            def write(rec):
                rec.update(material=material, scan_type=scan_type, phase=name, source="step6")
                runs.write(json.dumps(rec, sort_keys=True) + "\n")
                runs.flush()
                recs.append(rec)
                return rec

            wake = write(s5.run_one(phase_dir, case_id, n_probe, "wakeup_" + WAKE, WAKE, "wakeup", None))
            print(f"  wakeup {WAKE} status {wake['status']} rc {wake['rc']} seconds {wake['seconds']}",
                  file=log, flush=True)
            stock_s = None
            for m in order:
                rec = write(s5.run_one(phase_dir, case_id, n_probe, m, m, "timed", stock_s))
                if m == "stock" and rec["status"] == "ok":
                    stock_s = rec["seconds"]
                print(f"  timed {m} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']} n {rec['n']} "
                      f"limit {rec['limit_s']}", file=log, flush=True)
            timed = [r for r in recs if r["kind"] == "timed"]
            for r in timed:
                # brief rule, checked after the case: GPU limit = max(2 x stock s, 60 s)
                if r["method"] != "stock" and r["status"] == "ok" and stock_s is not None \
                        and r["seconds"] > max(2.0 * stock_s, 60.0):
                    print(f"  OVERRUN {r['method']} seconds {r['seconds']} > max(2 x stock, 60) "
                          f"{max(2.0 * stock_s, 60.0)}", file=log, flush=True)
            for a in accuracy_for_case(recs, material, scan_type, case_id):
                acc.write(json.dumps(a, sort_keys=True) + "\n")
                acc.flush()
                if a["method"] != "stock":
                    print(f"  accuracy {a['method']} max_d {a['max_d']} pass {a['pass']} "
                          f"err {a['accuracy_error']}", file=log, flush=True)
            all_recs[case_id] = (material, timed)
    finally:
        runs.close()
        acc.close()
    return all_recs


def reused_rows_polar():
    """The 5 copper polar cases from step 2 (runs_final8, 8 GPU timed rows each) and the reference stock rows."""
    rows = []
    final8_path = paths.RESULTS_DIR / "magma_host" / "runs_final8.jsonl"
    ref_path = paths.RESULTS_DIR / "reference" / "reference.jsonl"
    for p, producer in [(final8_path, "drivers/magma_host_merge.py (copy its runs_final8.jsonl here)"),
                        (ref_path, "src/time_stock.py (output folder RESULTS_DIR/reference)")]:
        if not p.is_file():
            raise SystemExit(f"error: {p} not found. Produce it first with {producer}")
    final8 = [json.loads(l) for l in final8_path.read_text().splitlines() if l.strip()]
    ref = [json.loads(l) for l in ref_path.read_text().splitlines() if l.strip()]
    for case_id in REUSED_POLAR:
        for r in final8:
            if r["kind"] == "timed" and r["case"] == case_id:
                rows.append(dict(source="magma_host/runs_final8.jsonl", material="copper", scan_type="polar",
                                 case=case_id, method=r["method"], status=r["status"], rc=r["rc"],
                                 seconds=r["seconds"], n=r["n"], stock_seconds=r.get("stock_seconds"),
                                 max_d=r["max_d"], run_dir=r["run_dir"]))
        for r in ref:
            if r["kind"] == "timed" and r["case"] == case_id:
                rows.append(dict(source="reference/reference.jsonl", material="copper", scan_type="polar",
                                 case=case_id, method="stock", status="ok" if r["returncode"] == 0 else "failed",
                                 rc=r["returncode"], seconds=r["seconds"], n=r["n"], stock_seconds=r["seconds"],
                                 max_d=None, run_dir=None))
    return rows


def write_all_polar(all_recs, acc_rows):
    acc_by = {(a["case"], a["method"]): a for a in acc_rows}
    reused = reused_rows_polar()
    path = OUT / "all_polar.jsonl"
    with path.open("x") as f:
        for r in reused:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        for case_id, (material, timed) in all_recs.items():
            for t in timed:
                a = acc_by[(case_id, t["method"])]
                f.write(json.dumps(dict(source="step6/polar_runs.jsonl", material=material, scan_type="polar",
                                        case=case_id, method=t["method"], status=t["status"], rc=t["rc"],
                                        seconds=t["seconds"], n=t["n"], stock_seconds=a["stock_seconds"],
                                        speedup=a["speedup"], max_d=a["max_d"], run_dir=t.get("run_dir")),
                                   sort_keys=True) + "\n")
    print("wrote", path, flush=True)


def main():
    if os.environ.get("TMPDIR") != str(rg.TMPDIR):
        raise SystemExit(f"TMPDIR must be {rg.TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if len(sys.argv) != 1:
        raise SystemExit(__doc__)
    os.environ["MPLCONFIGDIR"] = str(rg.TMPDIR / "mpl")
    (rg.TMPDIR / "mpl").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "polar_done.txt").exists() or (OUT / "stereo_done.txt").exists():
        raise SystemExit("done file already present; choose a new output folder")
    log = sys.stdout
    print("TMPDIR", rg.TMPDIR, flush=True)
    print("polar new cases:", [r["id"] for r in new_cases("polar", REUSED_POLAR)], flush=True)
    print("stereo new cases:", [r["id"] for r in new_cases("stereographic", REUSED_STEREO)], flush=True)
    status = "FINISHED"
    polar_recs = None
    try:
        polar_recs = run_phase("polar", "polar", "polar", POLAR_METHODS, REUSED_POLAR, log)
        acc_rows = [json.loads(l) for l in (OUT / "polar_accuracy.jsonl").read_text().splitlines() if l.strip()]
        write_all_polar(polar_recs, acc_rows)
        (OUT / "polar_done.txt").write_text("FINISHED\n")
        print("POLAR FINISHED", flush=True)
    except BaseException as e:
        (OUT / "polar_done.txt").write_text(f"ABORTED: {e!r}\n")
        (OUT / "stereo_done.txt").write_text("NOT STARTED: polar phase did not finish\n")
        print(f"POLAR ABORTED: {e!r}", flush=True)
        raise
    try:
        run_phase("stereo", "stereographic", "stereographic", STEREO_METHODS, REUSED_STEREO, log)
        (OUT / "stereo_done.txt").write_text("FINISHED\n")
        print("STEREO FINISHED", flush=True)
    except BaseException as e:
        (OUT / "stereo_done.txt").write_text(f"ABORTED: {e!r}\n")
        print(f"STEREO ABORTED: {e!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
