"""Step 5: stereographic check of the GPU variants on 5 copper stereo cases (one calculation per method per case).

usage: TMPDIR=$REPO/.work python3 -B drivers/step5_run.py <out_dir>
Per case: one untimed wake-up (own_unrefined, kind "wakeup"), then stock, own_refined, own_unrefined in the
order random.Random(SEED).shuffle (fresh generator per case, order printed to the log).
Stock = $MSSPEC_STOCK_ROOT with no V2 variables (the stock_cpu rung of drivers/step3b_run.py).
GPU runs = dev package ($MSSPEC_GPU_ROOT) with MSSPEC_V2_GPU_METHOD set and no other switches.
Time = whole child process (time.perf_counter around Popen and communicate), as in time_stock.py.
Stop limit: stock none. GPU runs and the wake-up get max(2 x stock seconds, 60 s) if stock already ran in
this case, else 3600 s. A run over its limit is killed by PID tree walk (rg.kill_tree) and recorded as timeout.
Memory rule: need = 16 n^2 bytes; if need > MemAvailable - 2 GiB the run is recorded as skipped and not started.
Outputs in <out_dir>: runs.jsonl (one line per run), <case>/<label>/ (result.npz, meta.json, stdout.txt,
stderr.txt), accuracy.jsonl (after all runs: max|d| of own_* against this session's stock result.npz, pass at 0.01),
done.txt (FINISHED or ABORTED: ...).
"""
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
import paths  # noqa: E402
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(HERE))
import time_stock as ts  # noqa: E402
import run_gpu_variants as rg  # noqa: E402
import step3b_run as s3  # noqa: E402

CASE_DIR = paths.CASES_DIR / "stereographic"
CASES = [
    "copper_hkl_111_radius_3A_energy_150eV_depth_1_scan_stereographic_step_2deg",
    "copper_hkl_111_radius_4.5A_energy_150eV_depth_1_scan_stereographic_step_2deg",
    "copper_hkl_111_radius_5.5A_energy_300eV_depth_1_scan_stereographic_step_2deg",
    "copper_hkl_111_radius_8.5A_energy_150eV_depth_1_scan_stereographic_step_2deg",
    "copper_hkl_111_radius_10.5A_energy_150eV_depth_1_scan_stereographic_step_2deg",
]
METHODS = ["stock", "own_refined", "own_unrefined"]
WAKE_METHOD = "own_unrefined"
GPU_EXPECT = (1, 1)  # gpu_build=1, early_start=1 with no switches (as the step 3b wake-up and rungs)
UNKNOWN_STOCK_LIMIT = 3600.0
TOL = rg.TOL


def spec(method_key):
    if method_key == "stock":
        return dict(package="stock", method=None, extra={}, expect=None)
    return dict(package="dev", method=method_key, extra={}, expect=GPU_EXPECT)


def write_row(log, rec):
    log.write(json.dumps(rec, sort_keys=True) + "\n")
    log.flush()


def run_one(out_root, case_id, n_probe, label, method_key, kind, stock_s):
    sp = spec(method_key)
    if sp["method"] is None:
        limit = None
    elif stock_s is not None:
        limit = max(2.0 * stock_s, 60.0)
    else:
        limit = UNKNOWN_STOCK_LIMIT
    rg.preflight()  # no other run_case.py process, else SystemExit (nothing is timed)
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    swap_before = s3.meminfo("SwapFree")
    need = 16 * n_probe ** 2
    rec = dict(kind=kind, case=case_id, n_probe=n_probe, label=label, method=sp["method"] or "stock",
               package=sp["package"], switches=sp["extra"], limit_s=limit, stock_ref_seconds=stock_s,
               busy_before=busy, mem_available_before_bytes=mem_before, swap_free_before_bytes=swap_before,
               need_bytes=need, preflight="no other run_case.py process")
    if need > mem_before - ts.RESERVE_BYTES:
        rec.update(status="skipped", rc=None, seconds=None, n=None,
                   reason=f"need {need} bytes > MemAvailable {mem_before} minus 2 GiB; not started")
        return rec
    case_path = (CASE_DIR / f"{case_id}.json").resolve()
    run_dir = out_root / case_id / label
    run_dir.mkdir(parents=True)
    scratch = rg.TMPDIR / "step5_cwd" / case_id / label
    scratch.mkdir(parents=True)
    cmd = [sys.executable, str(ts.SCRIPT), str(case_path), str(run_dir.resolve())]
    env = s3.child_env(sp["package"], sp["method"], sp["extra"])
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=str(scratch), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    timed_out, killed = False, []
    try:
        out, err = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        killed = rg.kill_tree(p.pid)
        out, err = p.communicate()
    seconds = time.perf_counter() - t0
    (run_dir / "stdout.txt").write_text(out)
    (run_dir / "stderr.txt").write_text(err)
    rc = None if timed_out else p.returncode
    lines = err.splitlines()
    proof_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU method=")]
    failed_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU_FAILED")]
    proof = s3.parse_proof(proof_lines[-1]) if proof_lines else {}
    if timed_out:
        status = "timeout"
    elif rc == 0 and (run_dir / "result.npz").exists():
        status = "ok"
    else:
        status = "failed"
    # Switch proof: the stock run must print no GPU line; a GPU run must print its own method and flags.
    if sp["method"] is None and proof_lines:
        raise SystemExit(f"{case_id} {label}: stock run printed a GPU proof line: {proof_lines[-1]}")
    if sp["method"] is not None and proof_lines:
        if proof.get("method") != sp["method"]:
            raise SystemExit(f"{case_id} {label}: proof method {proof.get('method')} != {sp['method']}")
        if (proof.get("gpu_build"), proof.get("early_start")) != sp["expect"]:
            raise SystemExit(f"{case_id} {label}: proof flags {proof.get('gpu_build')},{proof.get('early_start')}"
                             f" != expected {sp['expect']}")
    if status == "ok" and sp["method"] is not None and not proof_lines:
        raise SystemExit(f"{case_id} {label}: ok GPU run without a proof line")
    n = None
    if status == "ok":
        orders = json.loads((run_dir / "meta.json").read_text())["matrix_order"]
        if len(orders) != 1:
            raise SystemExit(f"{case_id} {label}: expected one matrix order, got {orders}")
        n = int(orders[0])
    rec.update(rc=rc, status=status, seconds=seconds, n=n, n_equals_probe=(n == n_probe) if n else None,
               timeout_killed_pids=killed, pythonpath=env["PYTHONPATH"], proof=proof,
               proof_line=proof_lines[-1] if proof_lines else None,
               failed_line=failed_lines[-1] if failed_lines else None,
               stderr_last_line=(err.strip().splitlines() or [None])[-1], run_dir=str(run_dir),
               mhz_after=ts.cpu_mhz(range(ts.THREADS)))
    return rec


def accuracy(out_root, rows):
    def find(case_id, method):
        hits = [r for r in rows if r["case"] == case_id and r["method"] == method and r["kind"] == "timed"]
        if len(hits) != 1:
            raise SystemExit(f"{case_id} {method}: {len(hits)} timed rows, expected 1")
        return hits[0]

    with (out_root / "accuracy.jsonl").open("w") as f:
        for case_id in CASES:
            st = find(case_id, "stock")
            for method in ["own_refined", "own_unrefined"]:
                row = find(case_id, method)
                rec = dict(case=case_id, method=method, n=row["n"], stock_status=st["status"],
                           method_status=row["status"], max_d=None, **{"pass": None}, theta_phi_identical=None)
                if st["status"] == "ok" and row["status"] == "ok":
                    test = Path(row["run_dir"]) / "result.npz"
                    ref = Path(st["run_dir"]) / "result.npz"
                    ac = subprocess.run([sys.executable, "-B", str(V2 / "src" / "check_accuracy.py"),
                                         str(test), str(ref)], capture_output=True, text=True)
                    (Path(row["run_dir"]) / "accuracy.txt").write_text(ac.stdout + ac.stderr)
                    m = re.search(r"max\|d\| over all blocks = (\S+?),", ac.stdout)
                    if m is None:
                        raise SystemExit(f"{case_id} {method}: no max|d| line:\n{ac.stdout}{ac.stderr}")
                    rec["max_d"] = float(m.group(1))
                    rec["pass"] = rec["max_d"] <= TOL
                    with np.load(test) as zt, np.load(ref) as zr:
                        keys = [k for k in zr.files if k.startswith(("theta_", "phi_"))]
                        rec["theta_phi_identical"] = bool(all(np.array_equal(zt[k], zr[k]) for k in keys))
                        rec["keys_test"] = zt.files
                        rec["keys_ref"] = zr.files
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                print(f"accuracy {case_id} {method} max_d {rec['max_d']} pass {rec['pass']} "
                      f"theta_phi_identical {rec['theta_phi_identical']}", flush=True)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    if os.environ.get("TMPDIR") != str(rg.TMPDIR):
        raise SystemExit(f"TMPDIR must be {rg.TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if (out_root / "runs.jsonl").exists() or (out_root / "done.txt").exists():
        raise SystemExit(f"{out_root} already has runs.jsonl or done.txt; choose a new output directory")
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(rg.TMPDIR / "mpl")
    print("TMPDIR", rg.TMPDIR, flush=True)
    print(f"free RAM before start: MemAvailable {s3.meminfo('MemAvailable')} bytes, "
          f"SwapFree {s3.meminfo('SwapFree')} bytes", flush=True)
    print("stop-limit rule: stock none; GPU and wake-up max(2 x stock s, 60 s) after stock, else 3600 s", flush=True)
    status_final = "FINISHED"
    rows = []
    log = (out_root / "runs.jsonl").open("a")
    try:
        for case_id in CASES:
            case = json.loads((CASE_DIR / f"{case_id}.json").read_text())
            n_probe = int(case["matrix_order_probe"])
            order = METHODS[:]
            random.Random(rg.SEED).shuffle(order)
            print(f"case {case_id} n_probe {n_probe} order {order}", flush=True)
            rec = run_one(out_root, case_id, n_probe, "wakeup_" + WAKE_METHOD, WAKE_METHOD, "wakeup", None)
            write_row(log, rec)
            rows.append(rec)
            print(f"  wakeup {WAKE_METHOD} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']}",
                  flush=True)
            stock_s = None
            for method_key in order:
                rec = run_one(out_root, case_id, n_probe, method_key, method_key, "timed", stock_s)
                if method_key == "stock" and rec["status"] == "ok":
                    stock_s = rec["seconds"]
                write_row(log, rec)
                rows.append(rec)
                print(f"  timed {method_key} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']} "
                      f"n {rec['n']}", flush=True)
        accuracy(out_root, rows)
    except BaseException as e:  # record the stop, then re-raise
        status_final = f"ABORTED: {e!r}"
        print(status_final, flush=True)
        raise
    finally:
        log.close()
        (out_root / "done.txt").write_text(status_final + "\n")
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
