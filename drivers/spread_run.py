"""Run-to-run spread: stock, own_refined and own_unrefined on 5 large silicon/copper/tungsten polar cases, 3 rounds.

usage: TMPDIR=$REPO/.work python3 -B drivers/spread_run.py [--plan]
--plan prints the case list (with matrix_order_probe n) and the run order, then exits.
Protocol: one untimed warm-up of each method on silicon n=20875 (recorded in warmup.jsonl, not timed-reported).
Then 3 rounds. Round case orders: r1 forward, r2 reversed, r3 a fixed third order. Method order inside each case
rotates per round: r1 stock, own_refined, own_unrefined; r2 own_unrefined, stock, own_refined;
r3 own_refined, own_unrefined, stock. 45 timed runs in total.
Env exactly as the timed runs: s3.child_env("stock", None, {}) for stock (OPENBLAS_NUM_THREADS=20, no taskset),
s3.child_env("dev", m, {}) for own_refined / own_unrefined.
Per run: wait_for_idle first; clock = whole child process (perf_counter around Popen + communicate).
Stop limit: stock none; GPU methods 3600 s (a run over it is killed by PID tree walk and recorded as timeout).
A run that needs more than MemAvailable - 2 GiB aborts the driver (the design needs every run).
Outputs in results/spread/: warmup/<method>/ and round<r>/<case>/<method>/ (meta.json, stdout.txt, stderr.txt,
result.npz only for the round-1 stock runs), warmup.jsonl, runs.jsonl (one line per timed run), done.txt on
FINISHED or ABORTED.txt on abort.
Reused by import, no copies: src/time_stock.py (SCRIPT, wait_for_idle, cpu_mhz, mem_available_bytes,
RESERVE_BYTES, THREADS), src/run_gpu_variants.py (TMPDIR, preflight, kill_tree), drivers/step3b_run.py
(child_env, parse_proof, meminfo).
"""
import hashlib
import json
import os
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

OUT = paths.RESULTS_DIR / "spread"
CASE_DIR = paths.CASES_DIR / "polar"
GPU_LIMIT_S = 3600.0
GPU_EXPECT = (1, 1)  # gpu_build=1, early_start=1 with no switches (same as step 5, step 6 and memory run)

# name -> (material prefix of the case id, expected matrix_order_probe)
CASE_SPECS = {
    "Si20875": ("silicon_hkl_001_", 20875),
    "Si26470": ("silicon_hkl_001_", 26470),
    "Si28383": ("silicon_hkl_001_", 28383),
    "Cu26741": ("copper_hkl_111_", 26741),
    "W27908": ("tungsten_hkl_110_", 27908),
}
WARMUP_CASE = "Si20875"
ROUND_CASES = {
    1: ["Si20875", "Si26470", "Si28383", "Cu26741", "W27908"],
    2: ["W27908", "Cu26741", "Si28383", "Si26470", "Si20875"],
    3: ["Si26470", "Cu26741", "Si20875", "W27908", "Si28383"],
}
ROUND_METHODS = {
    1: ["stock", "own_refined", "own_unrefined"],
    2: ["own_unrefined", "stock", "own_refined"],
    3: ["own_refined", "own_unrefined", "stock"],
}
WARMUP_METHODS = ["stock", "own_refined", "own_unrefined"]


def resolve_cases():
    files = {}
    for f in sorted(CASE_DIR.glob("*.json")):
        c = json.loads(f.read_text())
        if c["id"] != f.stem:
            raise SystemExit(f"case id {c['id']} != file name {f.stem}")
        files[f.stem] = int(c["matrix_order_probe"])
    out = {}
    for name, (prefix, n) in CASE_SPECS.items():
        hits = [cid for cid, probe in files.items() if cid.startswith(prefix) and probe == n]
        if len(hits) != 1:
            raise SystemExit(f"{name}: expected one case with prefix {prefix} and n={n}, got {hits}")
        out[name] = (hits[0], n)
    return out


def check_plan(cases):
    for r in (1, 2, 3):
        if sorted(ROUND_CASES[r]) != sorted(cases):
            raise SystemExit(f"round {r} case list does not cover all cases")
        if sorted(ROUND_METHODS[r]) != sorted(["stock", "own_refined", "own_unrefined"]):
            raise SystemExit(f"round {r} method list is not the 3 methods")


def read_cross_section(npz_path):
    with np.load(npz_path) as z:
        if "cross_section_0" not in z.files:
            raise SystemExit(f"{npz_path}: no cross_section_0 key, keys {z.files}")
        arr = np.ascontiguousarray(z["cross_section_0"])
    return hashlib.sha256(arr.tobytes()).hexdigest(), str(arr.dtype), list(arr.shape)


def run_one(tag, case_name, cases, method, run_dir, scratch, keep_npz):
    case_id, n_probe = cases[case_name]
    case_path = (CASE_DIR / f"{case_id}.json").resolve()
    package, mth = ("stock", None) if method == "stock" else ("dev", method)
    limit = None if method == "stock" else GPU_LIMIT_S
    need = 16 * n_probe ** 2
    rg.preflight()  # no other run_case.py process, else SystemExit (nothing is measured)
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    if need > mem_before - ts.RESERVE_BYTES:
        raise SystemExit(f"{tag} {case_name} {method}: need {need} bytes > MemAvailable {mem_before} minus 2 GiB")
    run_dir.mkdir(parents=True)
    scratch.mkdir(parents=True)
    env = s3.child_env(package, mth, {})
    child = [sys.executable, str(ts.SCRIPT), str(case_path), str(run_dir.resolve())]
    timed_out, killed = False, []
    t0 = time.perf_counter()
    p = subprocess.Popen(child, cwd=str(scratch), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    try:
        out, err = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        killed = rg.kill_tree(p.pid)
        out, err = p.communicate()
    wall = time.perf_counter() - t0
    (run_dir / "stdout.txt").write_text(out)
    (run_dir / "stderr.txt").write_text(err)
    rc = None if timed_out else p.returncode
    lines = err.splitlines()
    proof_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU method=")]
    failed_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU_FAILED")]
    proof = s3.parse_proof(proof_lines[-1]) if proof_lines else {}
    if method == "stock" and proof_lines:
        raise SystemExit(f"{tag} {case_name} {method}: stock run printed a GPU proof line: {proof_lines[-1]}")
    if method != "stock" and proof_lines:
        if proof.get("method") != method:
            raise SystemExit(f"{tag} {case_name} {method}: proof method {proof.get('method')} != {method}")
        if (proof.get("gpu_build"), proof.get("early_start")) != GPU_EXPECT:
            raise SystemExit(f"{tag} {case_name} {method}: proof flags {proof.get('gpu_build')},"
                             f"{proof.get('early_start')} != expected {GPU_EXPECT}")
    if timed_out:
        status = "timeout"
    elif rc == 0 and (run_dir / "result.npz").exists():
        status = "ok"
    elif rc == 3:
        status = "rc3"  # device out of memory, recorded and not retried
    else:
        status = "failed"
    if status == "ok" and method != "stock" and not proof_lines:
        raise SystemExit(f"{tag} {case_name} {method}: ok GPU run without a proof line")
    n = None
    sha = dtype = shape = None
    if status == "ok":
        orders = json.loads((run_dir / "meta.json").read_text())["matrix_order"]
        if len(orders) != 1:
            raise SystemExit(f"{tag} {case_name} {method}: expected one matrix order, got {orders}")
        n = int(orders[0])
        if n != n_probe:
            raise SystemExit(f"{tag} {case_name} {method}: matrix order {n} != probe {n_probe}")
        sha, dtype, shape = read_cross_section(run_dir / "result.npz")
        if not keep_npz:
            (run_dir / "result.npz").unlink()
    elif (run_dir / "result.npz").exists() and not keep_npz:
        (run_dir / "result.npz").unlink()
    rec = dict(round=tag, case=case_name, case_id=case_id, n=n, n_probe=n_probe, method=method, package=package,
               rc=rc, status=status, wall_s=wall, busy_before=busy, mhz_after=ts.cpu_mhz(range(ts.THREADS)),
               proof=proof, proof_line=proof_lines[-1] if proof_lines else None,
               failed_line=failed_lines[-1] if failed_lines else None, sha256_cross_section_0=sha,
               cross_section_0_dtype=dtype, cross_section_0_shape=shape, stop_limit_s=limit,
               timeout_killed_pids=killed, mem_available_before_bytes=mem_before, run_dir=str(run_dir),
               npz_kept=bool(keep_npz and status == "ok"))
    return rec


def main():
    cases = resolve_cases()
    check_plan(cases)
    if sys.argv[1:] == ["--plan"]:
        for name, (cid, n) in cases.items():
            print(f"case {name} n={n} {cid}")
        print("warm-up", WARMUP_CASE, WARMUP_METHODS)
        for r in (1, 2, 3):
            print(f"round {r} cases", ROUND_CASES[r], "methods per case", ROUND_METHODS[r])
        print("timed runs", 3 * len(CASE_SPECS) * 3)
        return
    if len(sys.argv) != 1:
        raise SystemExit(__doc__)
    if os.environ.get("TMPDIR") != str(rg.TMPDIR):
        raise SystemExit(f"TMPDIR must be {rg.TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    for name in ("runs.jsonl", "warmup.jsonl", "done.txt", "ABORTED.txt"):
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; choose a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    (rg.TMPDIR / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(rg.TMPDIR / "mpl")
    print("TMPDIR", rg.TMPDIR, flush=True)
    print("cases:", [(name, cid, n) for name, (cid, n) in cases.items()], flush=True)
    print(f"free RAM before start: MemAvailable {s3.meminfo('MemAvailable')} bytes, "
          f"SwapFree {s3.meminfo('SwapFree')} bytes", flush=True)
    status_final = "FINISHED"
    warm_log = (OUT / "warmup.jsonl").open("x")
    runs = None
    try:
        print("warm-up started", flush=True)
        for m in WARMUP_METHODS:
            rec = run_one("warmup", WARMUP_CASE, cases, m, OUT / "warmup" / m,
                          rg.TMPDIR / "spread_cwd" / "warmup" / m, keep_npz=False)
            warm_log.write(json.dumps(rec, sort_keys=True) + "\n")
            warm_log.flush()
            print(f"warm-up {WARMUP_CASE} {m} status {rec['status']} rc {rec['rc']} (untimed)", flush=True)
            if rec["status"] != "ok":
                raise SystemExit(f"warm-up {m} not ok: status {rec['status']}, rc {rec['rc']}")
        runs = (OUT / "runs.jsonl").open("x")
        for r in (1, 2, 3):
            for case_name in ROUND_CASES[r]:
                for m in ROUND_METHODS[r]:
                    tag = f"round{r}"
                    keep = (r == 1 and m == "stock")
                    rec = run_one(tag, case_name, cases, m, OUT / tag / cases[case_name][0] / m,
                                  rg.TMPDIR / "spread_cwd" / tag / cases[case_name][0] / m, keep_npz=keep)
                    runs.write(json.dumps(rec, sort_keys=True) + "\n")
                    runs.flush()
                    print(f"{tag} {case_name} n={rec['n']} {m} status {rec['status']} rc {rec['rc']} "
                          f"wall_s {rec['wall_s']:.3f} busy_before {rec['busy_before']} "
                          f"sha256 {rec['sha256_cross_section_0']}", flush=True)
    except BaseException as e:  # record the stop, then re-raise
        status_final = f"ABORTED: {e!r}"
        (OUT / "ABORTED.txt").write_text(status_final + "\n")
        print(status_final, flush=True)
        raise
    finally:
        warm_log.close()
        if runs is not None:
            runs.close()
        if status_final == "FINISHED":
            (OUT / "done.txt").write_text("FINISHED\n")
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
