"""Path configuration for the drivers and for the src/ scripts they call.

Repo-relative defaults; environment variables override them.

  MSSPEC_STOCK_ROOT  folder that contains msspec_python3/ (unmodified MsSpec). Needed for stock runs.
  MSSPEC_GPU_ROOT    folder that contains msspec_python3/ (MsSpec with patches/msspec_gpu.patch, built).
                     Needed for GPU runs.
  MSSPEC_GPU_LIB     the built GPU-enabled Fortran module (default: the phd_mi module inside MSSPEC_GPU_ROOT).
  RESULTS_DIR        new outputs of the drivers (default: REPO/results/runs).
  CASES_DIR          case files (default: REPO/cases).
  WORK_DIR           temporary files of the runs (default: REPO/.work, git-ignored). Set TMPDIR to this folder.

The two MsSpec roots are checked only when they are used (stock_pythonpath, gpu_pythonpath), so a driver
that does not run MsSpec can be imported without them.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dir(name, default):
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


CASES_DIR = _dir("CASES_DIR", REPO / "cases")
RESULTS_DIR = _dir("RESULTS_DIR", REPO / "results" / "runs")
WORK_DIR = _dir("WORK_DIR", REPO / ".work")


def _msspec_root(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"error: environment variable {name} is not set (see README, 'Running the calculations')")
    root = Path(value).expanduser().resolve()
    if not (root / "msspec_python3" / "src").is_dir():
        raise SystemExit(f"error: {name}={root} has no msspec_python3/src folder")
    return root


def stock_pythonpath():
    return _msspec_root("MSSPEC_STOCK_ROOT") / "msspec_python3" / "src"


def gpu_lib():
    value = os.environ.get("MSSPEC_GPU_LIB", "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return (_msspec_root("MSSPEC_GPU_ROOT") / "msspec_python3" / "src" / "msspec" / "spec" / "fortran"
            / "_phd_mi_noso_nosp_nosym.so")


def gpu_pythonpath():
    root = _msspec_root("MSSPEC_GPU_ROOT")
    lib = gpu_lib()
    if not lib.is_file():
        raise SystemExit(f"error: GPU library not found: {lib} (set MSSPEC_GPU_LIB, or build it with scripts/build_msspec_gpu.sh)")
    return root / "msspec_python3" / "src"
