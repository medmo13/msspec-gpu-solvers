"""Time the stock CPU reference ($MSSPEC_STOCK_ROOT, unmodified) on study cases, one timed run per case.

usage: python3 src/time_stock.py <output_dir> <case.json> [<case.json> ...]
The first case gets one untimed wake-up run first (kind "wakeup"). Then every case gets one timed run (kind "timed").
Each run starts only when the machine is idle (wait_for_idle from cpu_threads_sweep.py) and only if the case
needs less than MemAvailable - 2 GiB (need = 16 n^2 bytes, n = matrix_order_probe from the case file).
Time = whole child process (run_case.py), measured around the subprocess with time.perf_counter.
One JSON line per run is appended and flushed to <output_dir>/reference.jsonl.
The script stops at the first run with a non-zero return code.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drivers"))
import paths  # noqa: E402

ROOT = paths.REPO
SCRIPT = ROOT / "src" / "run" / "run_case.py"
sys.path.insert(0, str(ROOT / "src" / "run"))
from cpu_threads_sweep import cpu_mhz, wait_for_idle  # noqa: E402

THREADS = 20
RESERVE_BYTES = 2 * 1024 ** 3
DROP_PREFIXES = ("MSSPEC_SOLVER", "MSSPEC_V2_")


def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise SystemExit("MemAvailable not found in /proc/meminfo")


def child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROP_PREFIXES)}
    env.update(PYTHONPATH=str(paths.stock_pythonpath()), MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS=str(THREADS), PYTHONDONTWRITEBYTECODE="1")
    return env


def run_record(kind, case_path, out, label):
    case = json.loads(case_path.read_text())
    case_id, n_probe = case["id"], int(case["matrix_order_probe"])
    busy = wait_for_idle()
    mem_before = mem_available_bytes()
    need = 16 * n_probe ** 2
    if need > mem_before - RESERVE_BYTES:
        raise SystemExit(f"{case_id}: need {need} bytes > MemAvailable {mem_before} minus 2 GiB; stopped, no run")
    run_dir = out / label
    scratch = out / "scratch" / label
    scratch.mkdir(parents=True)
    cmd = [sys.executable, str(SCRIPT), str(case_path.resolve()), str(run_dir)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(scratch), env=child_env(), capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    ok = p.returncode == 0
    n = None
    sha = None
    if ok:
        orders = json.loads((run_dir / "meta.json").read_text())["matrix_order"]
        if len(orders) != 1:
            raise RuntimeError(f"{case_id}: expected one matrix order, got {orders}")
        n = int(orders[0])
        sha = hashlib.sha256((run_dir / "result.npz").read_bytes()).hexdigest()
    return dict(kind=kind, case=case_id, label=label, n=n, n_probe=n_probe, n_equals_probe=(n == n_probe),
                seconds=seconds, returncode=p.returncode, busy_before=busy, mhz_after=cpu_mhz(range(THREADS)),
                mem_available_before_bytes=mem_before, need_bytes=need, result_sha256=sha,
                stderr_tail=p.stderr[-500:], run_dir=str(run_dir))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    cases = [Path(a).resolve() for a in sys.argv[2:]]
    tmp = os.environ.get("TMPDIR", "")
    if not tmp or tmp.startswith("/tmp"):
        raise SystemExit(f"TMPDIR must be a folder inside the project, got {tmp!r}")
    print("TMPDIR", tmp, flush=True)
    log = out / "reference.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        rec = run_record("wakeup", cases[0], out, "wakeup_" + json.loads(cases[0].read_text())["id"])
        f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush()
        print("wakeup", rec["case"], "rc", rec["returncode"], "seconds", round(rec["seconds"], 2), flush=True)
        if rec["returncode"] != 0:
            raise SystemExit(f"wake-up failed, return code {rec['returncode']}: {rec['stderr_tail']}")
        for case_path in cases:
            rec = run_record("timed", case_path, out, json.loads(case_path.read_text())["id"])
            f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush()
            print("timed", rec["case"], "n", rec["n"], "probe", rec["n_probe"], "rc", rec["returncode"],
                  "seconds", round(rec["seconds"], 2), flush=True)
            if rec["returncode"] != 0:
                raise SystemExit(f"{rec['case']} failed, return code {rec['returncode']}: {rec['stderr_tail']}")


if __name__ == "__main__":
    main()
