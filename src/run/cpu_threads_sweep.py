"""CPU best search: run unchanged stock MsSpec (run_case.py) with different thread counts and core sets.

usage: python3 src/run/cpu_threads_sweep.py <case.json> <runs> <output.jsonl> [config_name ...]
Time = whole process (start of python to end of output), measured around the subprocess.
Each configuration gets one untimed warm-up run, then <runs> timed runs. Rounds are in random order.
A run starts only when the machine is idle (see wait_for_idle); otherwise the script stops with an error.
One JSON line per run: config, round, seconds, returncode, matrix order from meta.json, result.npz SHA-256,
busy fraction before, CPU MHz after.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drivers"))
import paths  # noqa: E402

ROOT = paths.REPO
SCRIPT = ROOT / "src" / "run" / "run_case.py"

PERFORMANCE_THREADS = list(range(0, 12))              # 6 performance cores, 2 hardware threads each
PERFORMANCE_PHYSICAL = [0, 2, 4, 6, 8, 10]            # one hardware thread per performance core
EFFICIENCY = list(range(12, 20))                      # 8 efficiency cores, 1 hardware thread each

# name: (cpu list for taskset or None, OPENBLAS_NUM_THREADS or None)
CONFIGS = {
    "default_unpinned_no_thread_setting": (None, None),
    "threads_01_one_performance_core": ([0], 1),
    "threads_04_performance_physical": (PERFORMANCE_PHYSICAL[:4], 4),
    "threads_06_performance_physical": (PERFORMANCE_PHYSICAL, 6),
    "threads_08_performance_hyperthreads": (list(range(0, 8)), 8),
    "threads_12_performance_hyperthreads": (PERFORMANCE_THREADS, 12),
    "threads_14_all_physical_cores": (PERFORMANCE_PHYSICAL + EFFICIENCY, 14),
    "threads_20_all_hardware_threads": (list(range(0, 20)), 20),
    "threads_08_efficiency_cores_only": (EFFICIENCY, 8),
}


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]            # total, idle + iowait


def wait_for_idle(max_busy=None, seconds=2.0, patience_s=900):
    """Return the busy fraction of the whole CPU over `seconds`; wait until it is below max_busy.
    max_busy defaults to 0.10 (the desktop alone uses 4 to 8 %); CPU_SWEEP_MAX_BUSY overrides it (only for correctness checks, never for timing)."""
    max_busy = float(os.environ.get("CPU_SWEEP_MAX_BUSY", 0.10)) if max_busy is None else max_busy
    start = time.time()
    while True:
        t0, i0 = cpu_times()
        time.sleep(seconds)
        t1, i1 = cpu_times()
        busy = 1.0 - (i1 - i0) / max(t1 - t0, 1)
        if busy <= max_busy:
            return busy
        if time.time() - start > patience_s:
            raise SystemExit(f"machine not idle after {patience_s} s (busy {busy:.3f}); no timing taken")


def cpu_mhz(cpus):
    out = []
    for c in cpus:
        try:
            out.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()) // 1000)
        except OSError:
            out.append(None)
    return out


def run_once(name, case_path):
    cpus, threads = CONFIGS[name]
    env = dict(os.environ, PYTHONPATH=str(paths.stock_pythonpath()), MPLBACKEND="Agg")
    for k in ("MSSPEC_SOLVER", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.pop(k, None)
    if threads is not None:
        env["OPENBLAS_NUM_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / "output"
        cmd = [sys.executable, str(SCRIPT), str(case_path), str(output)]
        if cpus is not None:
            cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
        busy = wait_for_idle()
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
        seconds = time.perf_counter() - t0
        meta = json.loads((output / "meta.json").read_text()) if p.returncode == 0 else None
        result_sha256 = hashlib.sha256((output / "result.npz").read_bytes()).hexdigest() if p.returncode == 0 else None
    return dict(config=name, seconds=seconds, returncode=p.returncode,
                matrix_order=meta["matrix_order"] if meta else None, result_sha256=result_sha256,
                stderr_tail=p.stderr[-300:], busy_before=busy, mhz_after=cpu_mhz(cpus or range(20)))


def checked_run(name, case_path):
    rec = run_once(name, case_path)
    if rec["returncode"] != 0:
        raise SystemExit(f"{name} failed (return code {rec['returncode']}): {rec['stderr_tail']}")
    return rec


def main():
    case_path, runs, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
    names = sys.argv[4:] or list(CONFIGS)
    random.seed(int(os.environ.get("CPU_SWEEP_SEED", 20261030)))   # a second batch of rounds sets its own seed
    with out.open("a") as f:
        for name in names:                                      # untimed warm-up, one per configuration
            w = checked_run(name, case_path)
            f.write(json.dumps(dict(w, kind="warmup", case=case_path.stem)) + "\n"); f.flush()
        for r in range(runs):
            order = names[:]
            random.shuffle(order)
            for name in order:
                rec = checked_run(name, case_path)
                f.write(json.dumps(dict(rec, kind="timed", round=r, case=case_path.stem)) + "\n"); f.flush()


if __name__ == "__main__":
    main()
