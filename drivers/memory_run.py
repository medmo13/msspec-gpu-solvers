"""Peak CPU RAM and peak GPU memory: stock + the 8 V2 GPU variants on 4 silicon polar cases (timings are not used).

usage: TMPDIR=$REPO/.work python3 -B drivers/memory_run.py [--plan]
--plan prints the case list (with matrix_order_probe n) and the 32 (case, method) rows in run order, then exits.
Reuse by import, no copies: src/time_stock.py (SCRIPT = src/run/run_case.py, wait_for_idle, mem_available_bytes,
RESERVE_BYTES), src/run_gpu_variants.py (TMPDIR, preflight, kill_tree), drivers/step3b_run.py (child_env,
parse_proof, meminfo), drivers/step6_run.py (POLAR_METHODS = stock + the 8 variants, in step 6 order).
Per run: one idle GPU query (gpu_idle_mib), then nvidia-smi -lms 20 sampler started BEFORE the child and stopped
BY PID after it; the child runs under /usr/bin/time -v -o <meter>/time_v.txt (max RSS of the child process).
Case selection: silicon polar cases in cases/polar, the one whose matrix_order_probe is closest to each target
(2500, 7500, 15850, 28383). Methods: the 3 smaller cases get stock + 8 variants (9 runs each); the largest gets
stock, magma_host_refined, magma_host_unrefined, own_refined, own_unrefined (5 runs). 32 runs in total.
Order per case: stock first, then the variants. A run needing more than MemAvailable - 2 GiB is recorded as skipped.
Stop limit (step 6 rule, timings not used): stock none; GPU max(2 x stock s, 60 s) if stock ran ok in this case,
else 3600 s. A run over its limit is killed by PID tree walk and recorded as timeout.
Outputs in results/memory/: runs.jsonl (one line per run), <case>/<method>/ (result.npz, meta.json, stdout.txt,
stderr.txt, accuracy.txt), <case>/<method>_meter/ (time_v.txt, gpu_samples.csv, nvidia_smi_stderr.txt), done.txt.
"""
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
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
import step6_run as s6  # noqa: E402

OUT = paths.RESULTS_DIR / "memory"
CASE_DIR = paths.CASES_DIR / "polar"
TIME_BIN = "/usr/bin/time"
SMI_BIN = "/usr/bin/nvidia-smi"
TARGETS = [2500, 7500, 15850, 28383]
LARGEST_METHODS = ["stock", "magma_host_refined", "magma_host_unrefined", "own_refined", "own_unrefined"]
GPU_EXPECT = (1, 1)  # gpu_build=1, early_start=1 with no switches (same as step 5 and step 6)


def silicon_polar_probes():
    rows = []
    for f in sorted(CASE_DIR.glob("silicon_hkl_001_*.json")):
        c = json.loads(f.read_text())
        if c["id"] != f.stem:
            raise SystemExit(f"case id {c['id']} != file name {f.stem}")
        rows.append((f.stem, int(c["matrix_order_probe"])))
    if not rows:
        raise SystemExit(f"no silicon polar cases in {CASE_DIR}")
    return rows


def build_plan():
    probes = silicon_polar_probes()
    cases = []
    for t in TARGETS:
        cid, n = min(probes, key=lambda r: (abs(r[1] - t), r[1]))
        cases.append((cid, n, t))
    plan = []
    for i, (cid, n, _t) in enumerate(cases):
        methods = list(s6.POLAR_METHODS) if i < 3 else list(LARGEST_METHODS)
        for m in methods:
            plan.append((cid, n, m))
    if len(plan) != 32:
        raise SystemExit(f"plan has {len(plan)} rows, expected 32")
    return cases, plan


def query_gpu():
    r = subprocess.run([SMI_BIN, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"nvidia-smi query failed, rc {r.returncode}: {r.stderr}")
    rows = [l for l in r.stdout.splitlines() if l.strip()]
    if len(rows) != 1:
        raise SystemExit(f"expected one GPU row from nvidia-smi, got {rows}")
    used, total = (int(x) for x in rows[0].split(","))
    return used, total


def read_samples(path):
    used = []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            if len(row) != 2:
                raise SystemExit(f"{path}: expected 2 columns, got {row}")
            used.append(int(row[0]))
    return used


def max_rss_kib(path):
    if not path.exists():
        return None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(m.group(1)) if m else None


def stop_limit(method, stock_s):
    if method == "stock":
        return None
    if stock_s is not None:
        return max(2.0 * stock_s, 60.0)
    return 3600.0


def run_one(case_id, n_probe, method, stock_s, stock_ok):
    package, mth = ("stock", None) if method == "stock" else ("dev", method)
    case_path = (CASE_DIR / f"{case_id}.json").resolve()
    run_dir = OUT / case_id / method
    meter = OUT / case_id / (method + "_meter")
    limit = stop_limit(method, stock_s)
    need = 16 * n_probe ** 2
    rg.preflight()  # no other run_case.py process, else SystemExit (nothing is measured)
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    swap_before = s3.meminfo("SwapFree")
    gpu_idle, gpu_total = query_gpu()
    rec = dict(case=case_id, n_probe=n_probe, n=None, n_equals_probe=None, method=method, package=package,
               switches={}, rc=None, status=None, wall_s=None, cpu_max_rss_kib=None, gpu_idle_mib=gpu_idle,
               gpu_peak_mib=None, gpu_total_mib=gpu_total, n_samples=None, max_d=None, **{"pass": None},
               accuracy_note=None, stop_limit_s=limit, stock_ref_seconds=stock_s, busy_before=busy,
               mem_available_before_bytes=mem_before, swap_free_before_bytes=swap_before,
               swap_free_after_bytes=None, need_bytes=need, run_dir=str(run_dir), proof=None, proof_line=None,
               failed_line=None, budget_lines=[], timeout_killed_pids=[], sampler_pid=None, sampler_rc=None,
               sampler_gone_after=None, time_v_path=None, gpu_samples_path=None)
    if need > mem_before - ts.RESERVE_BYTES:
        rec.update(status="skipped", reason=f"need {need} bytes > MemAvailable {mem_before} minus 2 GiB; not started")
        return rec
    run_dir.mkdir(parents=True)
    meter.mkdir(parents=True)
    scratch = rg.TMPDIR / "memory_cwd" / case_id / method
    scratch.mkdir(parents=True)
    env = s3.child_env(package, mth, {})
    child = [sys.executable, str(ts.SCRIPT), str(case_path), str(run_dir.resolve())]
    time_v = meter / "time_v.txt"
    cmd = [TIME_BIN, "-v", "-o", str(time_v)] + child
    samples_path = meter / "gpu_samples.csv"
    smi_out = samples_path.open("w")
    smi_err = (meter / "nvidia_smi_stderr.txt").open("w")
    sampler = subprocess.Popen([SMI_BIN, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                                "-lms", "20"], stdout=smi_out, stderr=smi_err)
    timed_out, killed = False, []
    try:
        if sampler.poll() is not None:
            raise SystemExit(f"{case_id} {method}: nvidia-smi sampler exited at start, rc {sampler.returncode}")
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=str(scratch), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        try:
            out, err = p.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            killed = rg.kill_tree(p.pid)
            out, err = p.communicate()
        wall = time.perf_counter() - t0
    finally:
        sampler_pid = sampler.pid
        sampler.send_signal(signal.SIGTERM)  # stop the sampler by its own PID
        sampler.wait(timeout=30)
        smi_out.close()
        smi_err.close()
        gone = not Path(f"/proc/{sampler_pid}").exists()
    (run_dir / "stdout.txt").write_text(out)
    (run_dir / "stderr.txt").write_text(err)
    rc = None if timed_out else p.returncode
    lines = err.splitlines()
    proof_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU method=")]
    failed_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU_FAILED")]
    budget_lines = [l for l in lines if l.startswith("MSSPEC_V2_GPU_BUDGET")]
    proof = s3.parse_proof(proof_lines[-1]) if proof_lines else {}
    # Switch proof, as in step 5 and step 6: stock prints no GPU line; a GPU run prints its own method and flags.
    if method == "stock" and proof_lines:
        raise SystemExit(f"{case_id} {method}: stock run printed a GPU proof line: {proof_lines[-1]}")
    if method != "stock" and proof_lines:
        if proof.get("method") != method:
            raise SystemExit(f"{case_id} {method}: proof method {proof.get('method')} != {method}")
        if (proof.get("gpu_build"), proof.get("early_start")) != GPU_EXPECT:
            raise SystemExit(f"{case_id} {method}: proof flags {proof.get('gpu_build')},{proof.get('early_start')}"
                             f" != expected {GPU_EXPECT}")
    if timed_out:
        status = "timeout"
    elif rc == 0 and (run_dir / "result.npz").exists():
        status = "ok"
    elif rc == 3:
        status = "rc3"  # device out of memory, recorded and not retried
    else:
        status = "failed"
    if status == "ok" and method != "stock" and not proof_lines:
        raise SystemExit(f"{case_id} {method}: ok GPU run without a proof line")
    n = n_probe
    if status == "ok":
        orders = json.loads((run_dir / "meta.json").read_text())["matrix_order"]
        if len(orders) != 1:
            raise SystemExit(f"{case_id} {method}: expected one matrix order, got {orders}")
        n = int(orders[0])
    samples = read_samples(samples_path)
    max_d = None
    if status == "ok" and method != "stock":
        if not stock_ok:
            rec["accuracy_note"] = "stock run of this case not ok; no comparison"
        else:
            ref = OUT / case_id / "stock" / "result.npz"
            ac = subprocess.run([sys.executable, "-B", str(V2 / "src" / "check_accuracy.py"),
                                 str(run_dir / "result.npz"), str(ref)], capture_output=True, text=True)
            (run_dir / "accuracy.txt").write_text(ac.stdout + ac.stderr)
            m = re.search(r"max\|d\| over all blocks = (\S+?),", ac.stdout)
            if m is None:
                raise SystemExit(f"{case_id} {method}: no max|d| line from check_accuracy.py:\n{ac.stdout}{ac.stderr}")
            max_d = float(m.group(1))
            rec["pass"] = max_d <= rg.TOL
    rec.update(rc=rc, status=status, n=n, n_equals_probe=(n == n_probe), wall_s=wall,
               cpu_max_rss_kib=max_rss_kib(time_v), gpu_peak_mib=(max(samples) if samples else None),
               n_samples=len(samples), max_d=max_d, timeout_killed_pids=killed,
               swap_free_after_bytes=s3.meminfo("SwapFree"), proof=proof,
               proof_line=proof_lines[-1] if proof_lines else None,
               failed_line=failed_lines[-1] if failed_lines else None, budget_lines=budget_lines,
               sampler_pid=sampler_pid, sampler_rc=sampler.returncode, sampler_gone_after=gone,
               time_v_path=str(time_v), gpu_samples_path=str(samples_path))
    return rec


def main():
    cases, plan = build_plan()
    if sys.argv[1:] == ["--plan"]:
        for cid, n, t in cases:
            print(f"case target={t} n={n} {cid}")
        for i, (cid, n, m) in enumerate(plan, 1):
            print(f"row {i:2d} n={n} {m} {cid}")
        print("rows", len(plan))
        return
    if len(sys.argv) != 1:
        raise SystemExit(__doc__)
    if os.environ.get("TMPDIR") != str(rg.TMPDIR):
        raise SystemExit(f"TMPDIR must be {rg.TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if (OUT / "runs.jsonl").exists() or (OUT / "done.txt").exists():
        raise SystemExit(f"{OUT} already has runs.jsonl or done.txt; choose a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    (rg.TMPDIR / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(rg.TMPDIR / "mpl")
    print("TMPDIR", rg.TMPDIR, flush=True)
    print("time:", TIME_BIN, "nvidia-smi:", SMI_BIN, flush=True)
    print("cases:", [(cid, n) for cid, n, _t in cases], flush=True)
    print(f"free RAM before start: MemAvailable {s3.meminfo('MemAvailable')} bytes, "
          f"SwapFree {s3.meminfo('SwapFree')} bytes", flush=True)
    stock_info = {}  # case -> (stock wall seconds, stock status ok)
    status_final = "FINISHED"
    runs = (OUT / "runs.jsonl").open("x")
    try:
        for case_id, n_probe, method in plan:
            s_sec, s_ok = stock_info.get(case_id, (None, False))
            rec = run_one(case_id, n_probe, method, s_sec, s_ok)
            if method == "stock":
                stock_info[case_id] = (rec["wall_s"], rec["status"] == "ok")
            runs.write(json.dumps(rec, sort_keys=True) + "\n")
            runs.flush()
            print(f"{case_id} n={rec['n']} {method} status {rec['status']} rc {rec['rc']} "
                  f"cpu_max_rss_kib {rec['cpu_max_rss_kib']} gpu_idle_mib {rec['gpu_idle_mib']} "
                  f"gpu_peak_mib {rec['gpu_peak_mib']} n_samples {rec['n_samples']} max_d {rec['max_d']}",
                  flush=True)
    except BaseException as e:  # record the stop, then re-raise
        status_final = f"ABORTED: {e!r}"
        print(status_final, flush=True)
        raise
    finally:
        runs.close()
        (OUT / "done.txt").write_text(status_final + "\n")
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
