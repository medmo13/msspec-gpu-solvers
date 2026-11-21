"""Step 2: one timed run of each GPU variant on the 5 copper polar cases, on the same clock as the stock reference.

usage: TMPDIR=$REPO/.work python3 src/run_gpu_variants.py <out_dir>
Reuses the helpers of src/time_stock.py by import (ts.wait_for_idle, ts.cpu_mhz, ts.mem_available_bytes,
ts.SCRIPT = src/run/run_case.py, ts.THREADS, ts.RESERVE_BYTES, ts.DROP_PREFIXES). Only the environment differs:
PYTHONPATH points to $MSSPEC_GPU_ROOT and MSSPEC_V2_GPU_METHOD is set.
Per case: one untimed wake-up run (own_refined, kind "wakeup"), then one timed run of each of the 6 variants in an
order shuffled with random.Random(20261118) (a fresh generator per case).
Time = whole child process, measured with time.perf_counter around the subprocess, as in time_stock.py.
Stop limit per run = max(2 x stock seconds, 60 s); a run over the limit is killed by PID (tree walk) and recorded as timeout.
One JSON line per run goes to <out_dir>/runs.jsonl. Files per run: <out_dir>/<case>/<label>/ (result.npz, meta.json,
stdout.txt, stderr.txt, accuracy.txt). <out_dir>/done.txt is written at the end (FINISHED or ABORTED: ...).
"""
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "drivers"))
import paths  # noqa: E402
import time_stock as ts  # noqa: E402

V2 = paths.REPO
REF_DIR = paths.RESULTS_DIR / "reference"
REF_JSONL = REF_DIR / "reference.jsonl"
CASE_DIR = paths.CASES_DIR / "polar"
TMPDIR = paths.WORK_DIR
METHODS = ["cusolver_refined", "cusolver_unrefined", "magma_refined", "magma_unrefined", "own_refined", "own_unrefined"]
WAKEUP_METHOD = "own_refined"
SEED = 20261118
TOL = 0.01
CASES = [
    "copper_hkl_111_radius_3A_energy_150eV_depth_1_scan_polar_step_1deg",
    "copper_hkl_111_radius_4.5A_energy_150eV_depth_1_scan_polar_step_1deg",
    "copper_hkl_111_radius_5.5A_energy_300eV_depth_1_scan_polar_step_1deg",
    "copper_hkl_111_radius_8.5A_energy_150eV_depth_1_scan_polar_step_1deg",
    "copper_hkl_111_radius_10.5A_energy_150eV_depth_1_scan_polar_step_1deg",
]
PROOF_KEYS = ("init_wait_s", "build_s", "lu_s", "refine_iters", "refine_s")


def load_stock():
    out = {}
    if not REF_JSONL.is_file():
        raise SystemExit(f"error: {REF_JSONL} not found. Produce it first with src/time_stock.py "
                         f"(output folder {REF_DIR}); see README, 'Running the calculations'")
    for line in REF_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] != "timed":
            continue
        if r["case"] in out:
            raise SystemExit(f"two timed stock rows for {r['case']}")
        if r["returncode"] != 0:
            raise SystemExit(f"stock row for {r['case']} has return code {r['returncode']}")
        out[r["case"]] = float(r["seconds"])
    missing = [c for c in CASES if c not in out]
    if missing:
        raise SystemExit(f"no stock timed row for {missing}")
    return out


def preflight():
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    me = os.getpid()
    hits = []
    for line in ps.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = int(parts[0]), parts[1]
        first = args.split()[0]
        if pid != me and "run_case.py" in args and "python" in first:
            hits.append(line.strip())
    if hits:
        raise SystemExit("other run_case.py process(es) running, no timing taken:\n" + "\n".join(hits))


def child_env(method):
    env = {k: v for k, v in os.environ.items() if not k.startswith(ts.DROP_PREFIXES)}
    env.update(PYTHONPATH=str(paths.gpu_pythonpath()), MPLBACKEND="Agg", OPENBLAS_NUM_THREADS=str(ts.THREADS),
               PYTHONDONTWRITEBYTECODE="1", MSSPEC_V2_GPU_METHOD=method, TMPDIR=str(TMPDIR))
    return env


def descendants(pid):
    ps = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = {}
    for line in ps.splitlines():
        f = line.split()
        if len(f) == 2:
            kids.setdefault(int(f[1]), []).append(int(f[0]))
    found, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            found.append(c)
            todo.append(c)
    return found


def kill_tree(pid):
    targets = [pid] + descendants(pid)
    for p in targets:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return targets


def run_one(out_root, case_id, method, kind, label, stock_s, n_probe):
    case_path = (CASE_DIR / f"{case_id}.json").resolve()
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    need = 16 * n_probe ** 2
    if need > mem_before - ts.RESERVE_BYTES:
        raise SystemExit(f"{case_id}: need {need} bytes > MemAvailable {mem_before} minus 2 GiB; stopped, no run")
    run_dir = out_root / case_id / label
    run_dir.mkdir(parents=True)
    scratch = TMPDIR / "step2_cwd" / case_id / label
    scratch.mkdir(parents=True)
    cmd = [sys.executable, str(ts.SCRIPT), str(case_path), str(run_dir.resolve())]
    limit = max(2.0 * stock_s, 60.0)
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=str(scratch), env=child_env(method), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    timed_out, killed = False, []
    try:
        out, err = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        killed = kill_tree(p.pid)
        out, err = p.communicate()
    seconds = time.perf_counter() - t0
    (run_dir / "stdout.txt").write_text(out)
    (run_dir / "stderr.txt").write_text(err)
    rc = None if timed_out else p.returncode
    lines = err.splitlines()
    proof_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU method=")]
    failed_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU_FAILED")]
    last_lines = [l for l in lines if l.strip()]
    if timed_out:
        status = "timeout"
    elif rc == 0 and (run_dir / "result.npz").exists():
        status = "ok"
    else:
        status = "failed"
    proof = {}
    if proof_lines:
        proof = {k: v for k, v in re.findall(r"(\w+)=(\S+)", proof_lines[-1])}
    rec = dict(case=case_id, n=n_probe, method=method, kind=kind, label=label, rc=rc, seconds=seconds,
               status=status, timeout_limit_s=limit, timeout_killed_pids=killed,
               failed_line=failed_lines[-1] if failed_lines else None,
               stderr_last_line=last_lines[-1] if last_lines else None,
               proof_method=proof.get("method"))
    for k in PROOF_KEYS:
        rec[k] = float(proof[k]) if k in proof and k != "refine_iters" else (
            int(proof[k]) if k in proof else None)
    rec.update(max_d=None, **{"pass": None}, stock_seconds=stock_s,
               speedup=(stock_s / seconds) if status == "ok" else None,
               busy_before=busy, mhz_after=ts.cpu_mhz(range(ts.THREADS)),
               mem_available_before_bytes=mem_before, need_bytes=need, n_meta=None, run_dir=str(run_dir))
    if status == "ok":
        meta = json.loads((run_dir / "meta.json").read_text())
        rec["n_meta"] = meta["matrix_order"]
        ac = subprocess.run([sys.executable, str(HERE / "check_accuracy.py"), str(run_dir / "result.npz"),
                             str(REF_DIR / case_id / "result.npz")], capture_output=True, text=True)
        (run_dir / "accuracy.txt").write_text(ac.stdout + ac.stderr)
        m = re.search(r"max\|d\| over all blocks = (\S+?),", ac.stdout)
        if m is None:
            raise SystemExit(f"{case_id} {label}: no max|d| line from check_accuracy.py:\n{ac.stdout}{ac.stderr}")
        rec["max_d"] = float(m.group(1))
        rec["pass"] = rec["max_d"] <= TOL
    return rec


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    if os.environ.get("TMPDIR") != str(TMPDIR):
        raise SystemExit(f"TMPDIR must be {TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if out_root.exists():
        raise SystemExit(f"{out_root} exists; choose a new output directory")
    stock = load_stock()
    preflight()
    out_root.mkdir(parents=True)
    print("TMPDIR", TMPDIR, flush=True)
    status_final = "FINISHED"
    log = (out_root / "runs.jsonl").open("a")
    try:
        for case_id in CASES:
            case = json.loads((CASE_DIR / f"{case_id}.json").read_text())
            n_probe = int(case["matrix_order_probe"])
            order = METHODS[:]
            random.Random(SEED).shuffle(order)
            print(f"case {case_id} n {n_probe} stock_seconds {stock[case_id]:.3f} order {order}", flush=True)
            rec = run_one(out_root, case_id, WAKEUP_METHOD, "wakeup", "wakeup_" + WAKEUP_METHOD,
                          stock[case_id], n_probe)
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
            print(f"  wakeup {WAKEUP_METHOD} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']:.3f}",
                  flush=True)
            for method in order:
                rec = run_one(out_root, case_id, method, "timed", method, stock[case_id], n_probe)
                log.write(json.dumps(rec, sort_keys=True) + "\n")
                log.flush()
                print(f"  timed {method} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']:.3f} "
                      f"max_d {rec['max_d']}", flush=True)
    except BaseException as e:  # record the stop, then re-raise
        status_final = f"ABORTED: {e!r}"
        print(status_final, flush=True)
        raise
    finally:
        log.close()
        (out_root / "done.txt").write_text(status_final + "\n")


if __name__ == "__main__":
    main()
