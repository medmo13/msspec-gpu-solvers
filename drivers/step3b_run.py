"""Step 3b: the ladder on the 5 copper polar cases, one timed run per rung, same clock as the step 2 runner.

usage: TMPDIR=$REPO/.work python3 -B drivers/step3b_run.py <out_dir>
Reuses by import (no edits to the step 2 runner): src/time_stock.py (wait_for_idle, cpu_mhz,
mem_available_bytes, RESERVE_BYTES, DROP_PREFIXES, SCRIPT, THREADS) and src/run_gpu_variants.py
(CASES, CASE_DIR, REF_DIR, REF_JSONL, paths.gpu_pythonpath, TMPDIR, SEED, TOL, load_stock, preflight, kill_tree).
Child environment is built here, per rung: stock_cpu uses the stock package ($MSSPEC_STOCK_ROOT) with
no V2 variables; the other rungs use the dev package ($MSSPEC_GPU_ROOT) and set only the switches of their rung.
Per case: one untimed wake-up (own_unrefined, no switches), then one timed run of each of the 6 rungs in an
order shuffled with random.Random(SEED) (a fresh generator per case, so every case has the same order).
Time = whole child process, time.perf_counter around the subprocess.
Stop limit per run = max(2 x stock reference seconds, 60 s); a run over the limit is killed by PID tree walk.
Output: <out_dir>/runs.jsonl (one line per run), <out_dir>/<case>/<label>/ (result.npz, meta.json, stdout.txt,
stderr.txt, accuracy.txt, run.log in calc/), <out_dir>/done.txt at the end (FINISHED or ABORTED: ...).
"""
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2 / "src"))
import paths  # noqa: E402
import time_stock as ts  # noqa: E402
import run_gpu_variants as rg  # noqa: E402

TMPDIR = rg.TMPDIR
# label, package, method (None = CPU path), switches, expected proof flags (gpu_build, early_start) or None
RUNGS = [
    ("stock_cpu", "stock", None, {}, None),
    ("cpu_rhs_reduction", "dev", None, {"MSSPEC_V2_RHS_REDUCTION": "1"}, None),
    ("gpu_solve_cpu_build", "dev", "own_unrefined", {"MSSPEC_V2_GPU_BUILD": "0", "MSSPEC_V2_EARLY_START": "0"},
     (0, 0)),
    ("gpu_build", "dev", "own_unrefined", {"MSSPEC_V2_EARLY_START": "0"}, (1, 0)),
    ("early_start", "dev", "own_unrefined", {}, (1, 1)),
    ("refinement", "dev", "own_refined", {}, (1, 1)),
]
WAKEUP = ("wakeup_own_unrefined", "dev", "own_unrefined", {}, (1, 1))


def meminfo(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise SystemExit(f"{key} not found in /proc/meminfo")


def parse_proof(line):
    out = {}
    for k, v in re.findall(r"(\w+)=(\S+)", line):
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
            continue
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def child_env(package, method, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(ts.DROP_PREFIXES)}
    pypath = paths.stock_pythonpath() if package == "stock" else paths.gpu_pythonpath()
    env.update(PYTHONPATH=str(pypath), MPLBACKEND="Agg", OPENBLAS_NUM_THREADS=str(ts.THREADS),
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(TMPDIR))
    if method is not None:
        env["MSSPEC_V2_GPU_METHOD"] = method
    env.update(extra)
    return env


def run_one(out_root, case_id, n_probe, label, package, method, extra, kind, stock_s, expect):
    limit = max(2.0 * stock_s, 60.0)
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    swap_before = meminfo("SwapFree")
    need = 16 * n_probe ** 2
    if need > mem_before - ts.RESERVE_BYTES:
        raise SystemExit(f"{case_id} {label}: need {need} bytes > MemAvailable {mem_before} minus 2 GiB; "
                         f"stopped, no run")
    case_path = (rg.CASE_DIR / f"{case_id}.json").resolve()
    run_dir = out_root / case_id / label
    run_dir.mkdir(parents=True)
    scratch = TMPDIR / "step3b_cwd" / case_id / label
    scratch.mkdir(parents=True)
    cmd = [sys.executable, str(ts.SCRIPT), str(case_path), str(run_dir.resolve())]
    env = child_env(package, method, extra)
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
    proof = parse_proof(proof_lines[-1]) if proof_lines else {}
    if timed_out:
        status = "timeout"
    elif rc == 0 and (run_dir / "result.npz").exists():
        status = "ok"
    elif rc == 3:
        status = "rc3"
    else:
        status = "failed"
    # Switch proof: a CPU rung must print no GPU line; a GPU rung must print its own method and flags.
    if method is None and proof_lines:
        raise SystemExit(f"{case_id} {label}: CPU rung printed a GPU proof line: {proof_lines[-1]}")
    if method is not None and proof_lines:
        if proof.get("method") != method:
            raise SystemExit(f"{case_id} {label}: proof method {proof.get('method')} != {method}")
        if expect is not None and (proof.get("gpu_build"), proof.get("early_start")) != expect:
            raise SystemExit(f"{case_id} {label}: proof flags {proof.get('gpu_build')},{proof.get('early_start')}"
                             f" != expected {expect}")
    if status == "ok" and method is not None and not proof_lines:
        raise SystemExit(f"{case_id} {label}: ok GPU run without a proof line")
    n = n_probe
    max_d, passed, sha = None, None, None
    if status == "ok":
        meta = json.loads((run_dir / "meta.json").read_text())
        orders = meta["matrix_order"]
        if len(orders) != 1:
            raise SystemExit(f"{case_id} {label}: expected one matrix order, got {orders}")
        n = int(orders[0])
        sha = hashlib.sha256((run_dir / "result.npz").read_bytes()).hexdigest()
        ac = subprocess.run([sys.executable, "-B", str(rg.V2 / "src" / "check_accuracy.py"),
                             str(run_dir / "result.npz"), str(rg.REF_DIR / case_id / "result.npz")],
                            capture_output=True, text=True)
        (run_dir / "accuracy.txt").write_text(ac.stdout + ac.stderr)
        m = re.search(r"max\|d\| over all blocks = (\S+?),", ac.stdout)
        if m is None:
            raise SystemExit(f"{case_id} {label}: no max|d| line from check_accuracy.py:\n{ac.stdout}{ac.stderr}")
        max_d = float(m.group(1))
        passed = max_d <= rg.TOL
    rec = dict(kind=kind, case=case_id, n=n, n_probe=n_probe, rung=label, package=package, method=method,
               switches=extra, rc=rc, status=status, seconds=seconds, timeout_limit_s=limit,
               timeout_killed_pids=killed, stock_ref_seconds=stock_s, proof=proof,
               proof_line=proof_lines[-1] if proof_lines else None,
               failed_line=failed_lines[-1] if failed_lines else None,
               max_d=max_d, **{"pass": passed}, result_sha256=sha, busy_before=busy,
               mhz_after=ts.cpu_mhz(range(ts.THREADS)), mem_available_before_bytes=mem_before,
               swap_free_before_bytes=swap_before, need_bytes=need, run_dir=str(run_dir),
               run_log=str(run_dir / "calc" / "run.log"))
    return rec


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    if os.environ.get("TMPDIR") != str(TMPDIR):
        raise SystemExit(f"TMPDIR must be {TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if (out_root / "runs.jsonl").exists() or (out_root / "done.txt").exists():
        raise SystemExit(f"{out_root} already has runs.jsonl or done.txt; choose a new output directory")
    stock = rg.load_stock()
    rg.preflight()
    out_root.mkdir(parents=True, exist_ok=True)
    print("TMPDIR", TMPDIR, flush=True)
    print(f"free RAM before start: MemAvailable {meminfo('MemAvailable')} bytes, "
          f"SwapFree {meminfo('SwapFree')} bytes", flush=True)
    print("stop-limit rule: max(2 x stock reference seconds, 60 s)", flush=True)
    print("rung order per case is random.Random(%d).shuffle(rung labels)" % rg.SEED, flush=True)
    status_final = "FINISHED"
    log = (out_root / "runs.jsonl").open("a")
    try:
        for case_id in rg.CASES:
            case = json.loads((rg.CASE_DIR / f"{case_id}.json").read_text())
            n_probe = int(case["matrix_order_probe"])
            labels = [r[0] for r in RUNGS]
            random.Random(rg.SEED).shuffle(labels)
            by_label = {r[0]: r for r in RUNGS}
            print(f"case {case_id} n {n_probe} stock_seconds {stock[case_id]:.3f} order {labels}", flush=True)
            w = WAKEUP
            rec = run_one(out_root, case_id, n_probe, w[0], w[1], w[2], w[3], "wakeup", stock[case_id], w[4])
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
            print(f"  wakeup {w[0]} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']:.3f}",
                  flush=True)
            for label in labels:
                _, package, method, extra, expect = by_label[label]
                rec = run_one(out_root, case_id, n_probe, label, package, method, extra, "timed", stock[case_id],
                              expect)
                log.write(json.dumps(rec, sort_keys=True) + "\n")
                log.flush()
                print(f"  timed {label} status {rec['status']} rc {rec['rc']} seconds {rec['seconds']:.3f} "
                      f"max_d {rec['max_d']}", flush=True)
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
