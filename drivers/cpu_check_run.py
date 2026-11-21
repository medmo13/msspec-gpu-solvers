"""CPU thread check: stock MsSpec (unchanged) at 4 CPU configurations on 2 large polar cases.

usage: TMPDIR=$REPO/.work python3 -B drivers/cpu_check_run.py <out_dir>
Reuses by import (no edits to any reused file):
  drivers/step3b_run.py  child_env("stock", None, {})  -> stock package, no V2 variables, OPENBLAS_NUM_THREADS=20
  src/time_stock.py    wait_for_idle, cpu_mhz, mem_available_bytes, RESERVE_BYTES, SCRIPT (src/run/run_case.py)
  src/run_gpu_variants.py  preflight (no other run_case.py running), TMPDIR, CASE_DIR
Pinning follows src/run/cpu_threads_sweep.py: PERFORMANCE_PHYSICAL = one logical CPU per P-core,
PERFORMANCE_THREADS = both logical CPUs of each P-core, EFFICIENCY = the 8 E-cores.
Each config sets OPENBLAS_NUM_THREADS to its thread count (the 20 of child_env is overridden for 6, 12, 14).

Protocol: one untimed wake-up per config on the smaller case (silicon), then per case one timed run per config.
Timed order: copper 06, 20, 12, 14; silicon 20, 14, 06, 12.
wait_for_idle before each run. Time = whole child process (subprocess.run), time.perf_counter around it.
Output: <out_dir>/runs.jsonl (one line per run), <out_dir>/<case>/<kind>_<config>/ (result.npz, meta.json,
calc/, stdout.txt, stderr.txt), <out_dir>/driver.log (stdout), <out_dir>/done.txt at the end.
The driver stops at the first run with a non-zero return code.
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
import step3b_run as s3  # noqa: E402
import run_gpu_variants as rg  # noqa: E402

COPPER = "copper_hkl_111_radius_10.5A_energy_150eV_depth_1_scan_polar_step_1deg"
SILICON = "silicon_hkl_001_radius_10.5A_energy_150eV_depth_1_scan_polar_step_1deg"

P_PHYSICAL = [0, 2, 4, 6, 8, 10]          # one logical CPU per P-core (PERFORMANCE_PHYSICAL in cpu_threads_sweep.py)
P_ALL = list(range(12))                   # both logical CPUs of the 6 P-cores (PERFORMANCE_THREADS)
E_CORES = list(range(12, 20))             # 8 E-cores, one logical CPU each (EFFICIENCY)

# name, taskset CPU list (None = no taskset), OPENBLAS_NUM_THREADS
CONFIGS = [
    ("threads_20_default", None, 20),
    ("threads_06_performance_physical", P_PHYSICAL, 6),
    ("threads_12_performance_all", P_ALL, 12),
    ("threads_14_physical", P_PHYSICAL + E_CORES, 14),
]
BY_NAME = {c[0]: c for c in CONFIGS}

WAKEUP_CASE = SILICON
WAKEUP_ORDER = ["threads_20_default", "threads_14_physical", "threads_06_performance_physical",
                "threads_12_performance_all"]
TIMED_ORDER = [
    (COPPER, ["threads_06_performance_physical", "threads_20_default", "threads_12_performance_all",
              "threads_14_physical"]),
    (SILICON, ["threads_20_default", "threads_14_physical", "threads_06_performance_physical",
               "threads_12_performance_all"]),
]


def case_path(case_id):
    return (rg.CASE_DIR / f"{case_id}.json").resolve()


def probe_of(case_id):
    return int(json.loads(case_path(case_id).read_text())["matrix_order_probe"])


def taskset_text(cpus):
    return "none (no taskset)" if cpus is None else ",".join(map(str, cpus))


def read_sys(cpu, rel):
    return Path(f"/sys/devices/system/cpu/cpu{cpu}/{rel}").read_text().strip()


def check_topology():
    """Confirm from sysfs which logical CPUs are P-cores (max freq >= 4.5 GHz) and that the lists are right."""
    lines = []
    for c in range(20):
        fmax = int(read_sys(c, "cpufreq/cpuinfo_max_freq")) // 1000
        kind = "P" if fmax >= 4500 else "E"
        lines.append(f"cpu{c}: {kind} max_mhz={fmax} core_id={read_sys(c, 'topology/core_id')} "
                     f"siblings={read_sys(c, 'topology/thread_siblings_list')}")
        if (kind == "P") != (c in P_ALL):
            raise SystemExit(f"topology check failed: cpu{c} kind {kind} but P_ALL is {P_ALL}")
    for c in P_PHYSICAL:
        sib = {int(x) for x in read_sys(c, "topology/thread_siblings_list").replace("-", ",").split(",")}
        if not sib <= set(P_ALL):
            raise SystemExit(f"topology check failed: P core cpu{c} siblings {sib} not inside P_ALL")
    for c in E_CORES:
        if read_sys(c, "topology/thread_siblings_list") != str(c):
            raise SystemExit(f"topology check failed: E cpu{c} has SMT siblings")
    return lines


def emit(log, rec):
    log.write(json.dumps(rec, sort_keys=True) + "\n")
    log.flush()
    print(f"  {rec['kind']} {rec['case']} {rec['config']} rc {rec['rc']} wall_s {rec['wall_s']:.3f} "
          f"n {rec['n']} sha256 {rec['cross_section_0_sha256']}", flush=True)
    if rec["rc"] != 0:
        raise RuntimeError(f"{rec['case']} {rec['config']} failed, rc {rec['rc']}: {rec['stderr_tail']}")


def run_one(kind, case_id, name, out_root):
    _, cpus, threads = BY_NAME[name]
    probe = probe_of(case_id)
    need = 16 * probe ** 2
    print(f"WAIT idle before {kind} {case_id} {name} time={time.strftime('%H:%M:%S')}", flush=True)
    busy = ts.wait_for_idle()
    mem_before = ts.mem_available_bytes()
    if need > mem_before - ts.RESERVE_BYTES:
        raise SystemExit(f"{case_id} {name}: need {need} bytes > MemAvailable {mem_before} minus 2 GiB; no run")
    run_dir = out_root / case_id / f"{kind}_{name}"
    scratch = rg.TMPDIR / "cpu_check_cwd" / case_id / f"{kind}_{name}"
    run_dir.mkdir(parents=True)
    scratch.mkdir(parents=True)
    env = s3.child_env("stock", None, {})
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    cmd = [sys.executable, str(ts.SCRIPT), str(case_path(case_id)), str(run_dir.resolve())]
    if cpus is not None:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    print(f"START {kind} {case_id} {name} taskset={taskset_text(cpus)} OPENBLAS_NUM_THREADS={threads} "
          f"time={time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(scratch), env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    (run_dir / "stdout.txt").write_text(p.stdout)
    (run_dir / "stderr.txt").write_text(p.stderr)
    rec = dict(kind=kind, case=case_id, config=name, taskset=taskset_text(cpus), threads=threads,
               openblas_num_threads=env["OPENBLAS_NUM_THREADS"], n_probe=probe, n=None, rc=p.returncode,
               wall_s=wall, busy_before=busy, mhz_after=ts.cpu_mhz(cpus if cpus is not None else range(20)),
               mem_available_before_bytes=mem_before, need_bytes=need, cross_section_0_sha256=None,
               stderr_tail=p.stderr[-300:], run_dir=str(run_dir))
    if p.returncode == 0:
        meta = json.loads((run_dir / "meta.json").read_text())
        orders = meta["matrix_order"]
        if len(orders) != 1:
            raise SystemExit(f"{case_id} {name}: expected one matrix order, got {orders}")
        rec["n"] = int(orders[0])
        rec["n_equals_probe"] = rec["n"] == probe
        with np.load(run_dir / "result.npz") as z:
            cs = z["cross_section_0"]
        rec["cross_section_0_sha256"] = hashlib.sha256(np.ascontiguousarray(cs).tobytes()).hexdigest()
    return rec


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    if os.environ.get("TMPDIR") != str(rg.TMPDIR):
        raise SystemExit(f"TMPDIR must be {rg.TMPDIR}, got {os.environ.get('TMPDIR')!r}")
    if (out_root / "runs.jsonl").exists() or (out_root / "done.txt").exists():
        raise SystemExit(f"{out_root} already has runs.jsonl or done.txt; choose a new output directory")
    out_root.mkdir(parents=True, exist_ok=True)
    print("TMPDIR", rg.TMPDIR, flush=True)
    print("OPENBLAS/stock env from step3b_run.child_env('stock', None, {}); package", paths.stock_pythonpath(),
          flush=True)
    print("== lscpu -e ==", flush=True)
    print(subprocess.run(["lscpu", "-e"], capture_output=True, text=True).stdout, end="", flush=True)
    print("== topology check (sysfs) ==", flush=True)
    for line in check_topology():
        print(" ", line, flush=True)
    print("== configurations ==", flush=True)
    for name, cpus, threads in CONFIGS:
        print(f"  {name}: taskset -c {taskset_text(cpus)}  OPENBLAS_NUM_THREADS={threads}  "
              f"logical CPUs={0 if cpus is None else len(cpus)}", flush=True)
    wake_probe = probe_of(WAKEUP_CASE)
    for case_id, _ in TIMED_ORDER:
        print(f"case {case_id} n_probe {probe_of(case_id)}", flush=True)
        if probe_of(case_id) <= wake_probe and case_id != WAKEUP_CASE:
            raise SystemExit("wake-up case is not the smaller case")
    print("wake-up order", WAKEUP_ORDER, "on", WAKEUP_CASE, flush=True)
    for case_id, order in TIMED_ORDER:
        print("timed order", case_id, order, flush=True)
    rg.preflight()
    status_final = "FINISHED"
    log = (out_root / "runs.jsonl").open("a")
    try:
        for name in WAKEUP_ORDER:
            emit(log, run_one("wakeup", WAKEUP_CASE, name, out_root))
        for case_id, order in TIMED_ORDER:
            for name in order:
                emit(log, run_one("timed", case_id, name, out_root))
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
