# msspec-gpu-solvers

GPU solvers for the matrix-inversion step of MsSpec, a multiple-scattering
code for electron diffraction. The repository holds eight mixed-precision
variants of the solve: cuSOLVER, MAGMA device, MAGMA host and an FP16-storage
LU, each refined and unrefined.

## Folder map

- `src/solver/` - the GPU solver source (`msspec_v2_gpu.cu`, `half_lu.cuh`).
- `patches/` - `msspec_gpu.patch`, the change to the stock MsSpec source that
  adds the GPU solver (see below).
- `scripts/` - `build_msspec_gpu.sh`, which builds MsSpec with the GPU solver.
- `drivers/` - run drivers (Python). They import helpers from `src/` and read
  the case files from `cases/`.
- `src/` - run helpers (`src/run/run_case.py`), the case writers, accuracy
  checks and plotting scripts.
- `cases/` - case definitions (JSON) and the case manifest (CSV).
- `LICENSE` - GNU General Public License v3.0.

## Apply the MsSpec patch

MsSpec itself is not included. Get it from its own project (GPL-3.0), then
apply the patch in the root of that checkout:

    cd /path/to/msspec_checkout
    patch -p1 < /path/to/msspec-gpu-solvers/patches/msspec_gpu.patch

The patch changes only source files (`.f`, `.f90`, `.py`, `.c`, `.cu`, `.h`,
`.mk`, `Makefile`, `meson.build`). Build outputs are not included.

## Running the calculations

The drivers in `drivers/` read their paths from `drivers/paths.py`. Set these
environment variables (only the ones your run needs):

- `MSSPEC_STOCK_ROOT` - folder that contains `msspec_python3/` for unmodified
  MsSpec. Needed for stock runs.
- `MSSPEC_GPU_ROOT` - folder that contains `msspec_python3/` for MsSpec with
  `patches/msspec_gpu.patch` applied and built. Needed for GPU runs.
- `MSSPEC_GPU_LIB` - the built GPU-enabled Fortran module. Default:
  `$MSSPEC_GPU_ROOT/msspec_python3/src/msspec/spec/fortran/_phd_mi_noso_nosp_nosym.so`.
- `RESULTS_DIR` - where new results are written. Default: `results/runs`.
- `CASES_DIR` - case files. Default: `cases`.
- `WORK_DIR` - temporary files of the runs. Default: `.work` (git-ignored).
  Set `TMPDIR` to the same folder; the drivers stop if it differs.
- `CUDA_HOME`, `MAGMA_ROOT` - used only by the build script below.

Build the GPU tree from a stock MsSpec checkout. The script copies the
checkout, applies the patch, points the copied Makefile at this repository's
`src/solver/msspec_v2_gpu.cu` and at nvcc, and runs the Fortran build:

    CUDA_HOME=/usr/local/cuda MAGMA_ROOT=/opt/magma \
      scripts/build_msspec_gpu.sh /path/to/stock_msspec /path/to/msspec_gpu

Some drivers read output of an earlier driver from `$RESULTS_DIR`. They stop
with a message if the file is missing:

- `src/time_stock.py <output_dir> <case.json> ...` writes
  `reference.jsonl` into `$RESULTS_DIR/reference/`. The GPU drivers
  (`src/run_gpu_variants.py`, `drivers/step6_run.py`) read it, together with
  the stock `result.npz` of each case in the same folder.
- `drivers/magma_host_merge.py` reads `$RESULTS_DIR/step2/runs_final.jsonl`.
- `drivers/step6_run.py` reads `$RESULTS_DIR/magma_host/runs_final8.jsonl`
  (the `runs_final8.jsonl` written by `drivers/magma_host_merge.py`).

## Examples

- `cases/polar/` - 30 polar-scan case files (JSON) for copper (111), silicon
  (001) and tungsten (110).
- `cases/stereographic/` - 30 stereographic-scan case files for the same
  three materials.
- `cases/manifest.csv` - one row per case (60 rows). Columns: `id`, `material`,
  `orientation`, `energy_eV`, `radius_A`, `atoms`, `n`, `target`,
  `relative_miss`, `scan_type`, `number_of_angles`, `memory_16n2_GiB`,
  `sha256`.
- `cases/probe_grid.csv` - probe values per material, energy, radius and
  depth (101 rows).

Smoke test: one copper polar case (n = 1210, radius 3 A), once with the stock
code and once with the GPU variant `own_unrefined`. Each run writes
`result.npz` and `meta.json` into `.work/smoke/`:

    export MSSPEC_STOCK_ROOT=/path/to/stock_msspec
    export MSSPEC_GPU_ROOT=/path/to/msspec_gpu
    CASE=cases/polar/copper_hkl_111_radius_3A_energy_150eV_depth_1_scan_polar_step_1deg.json
    mkdir -p .work/smoke
    PYTHONPATH="$MSSPEC_STOCK_ROOT/msspec_python3/src" python3 -B src/run/run_case.py "$CASE" .work/smoke/stock
    PYTHONPATH="$MSSPEC_GPU_ROOT/msspec_python3/src" MSSPEC_V2_GPU_METHOD=own_unrefined \
      python3 -B src/run/run_case.py "$CASE" .work/smoke/own_unrefined

## Requirements to run the code (not included)

- NVIDIA CUDA toolkit with cuSOLVER.
- MAGMA (device and host builds).
- MsSpec, patched as described above.
- Paths of MsSpec and of the run outputs come from environment variables;
  see "Running the calculations".

## Citation

Citation: see `CITATION.cff`. DOI: [10.5281/zenodo.24426080](https://doi.org/10.5281/zenodo.24426080) (version 1.0.0); all versions: 10.5281/zenodo.24426079

## Licence

GNU General Public License v3.0. See `LICENSE`.
