#!/usr/bin/env bash
# Build a GPU-enabled MsSpec tree (the MSSPEC_GPU_ROOT of the drivers) from a stock MsSpec checkout.
#
# usage: CUDA_HOME=/usr/local/cuda [MAGMA_ROOT=/opt/magma] scripts/build_msspec_gpu.sh <stock_root> <gpu_root_out>
#   <stock_root>    folder that contains msspec_python3/ (unmodified MsSpec). It is copied, never changed.
#   <gpu_root_out>  new folder; must not exist yet.
#   CUDA_HOME       CUDA toolkit folder (must contain bin/nvcc). Required.
#   MAGMA_ROOT      MAGMA folder (include/, lib/). Optional; if unset, the system MAGMA is used.
#
# Steps: copy the stock tree; apply patches/msspec_gpu.patch (-p1); point the copied spec Makefile at this
# repository's src/solver/msspec_v2_gpu.cu and at nvcc; build the phagen and spec Fortran modules with
# "make all" with PYTHON=python3 NO_VENV=1 (as the study's magma_host_build.sh does). The GPU solver is one object (msspec_v2_gpu.o) that is linked
# into _phd_mi_noso_nosp_nosym.so. Compile flags are the study's: nvcc -arch=sm_86 -O2 -Xcompiler=-fPIC.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: CUDA_HOME=... [MAGMA_ROOT=...] $0 <stock_root> <gpu_root_out>" >&2
    exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOCK="$(cd "$1" && pwd)"
OUT="$2"
[ -d "$STOCK/msspec_python3/src" ] || { echo "error: $STOCK has no msspec_python3/src" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "error: $OUT already exists" >&2; exit 1; }
: "${CUDA_HOME:?error: set CUDA_HOME to the CUDA toolkit folder (it must contain bin/nvcc)}"
NVCC="$CUDA_HOME/bin/nvcc"
[ -x "$NVCC" ] || { echo "error: nvcc not found at $NVCC" >&2; exit 1; }

GPU_NVCC_FLAGS="-arch=sm_86 -O2 -Xcompiler=-fPIC"
MAGMA_LINK_FLAGS="-L$CUDA_HOME/lib64"
if [ -n "${MAGMA_ROOT:-}" ]; then
    GPU_NVCC_FLAGS="$GPU_NVCC_FLAGS -I$MAGMA_ROOT/include"
    MAGMA_LINK_FLAGS="$MAGMA_LINK_FLAGS -L$MAGMA_ROOT/lib"
fi
# make reads these from the environment (the copied .mk uses $(NVCC), $(GPU_NVCC_FLAGS), $(MAGMA_LINK_FLAGS)).
export NVCC GPU_NVCC_FLAGS MAGMA_LINK_FLAGS

echo "[1/5] copy stock tree -> $OUT"
cp -a "$STOCK" "$OUT"

echo "[2/5] apply patches/msspec_gpu.patch"
patch -p1 -d "$OUT" < "$REPO/patches/msspec_gpu.patch"

F_PHAGEN="$OUT/msspec_python3/src/msspec/phagen/fortran"
F_SPEC="$OUT/msspec_python3/src/msspec/spec/fortran"
MK="$F_SPEC/phd_mi_noso_nosp_nosym.mk"

echo "[3/5] point the copied spec Makefile at this repository and at nvcc"
sed -i \
    -e "s|\.\./\.\./\.\./\.\./\.\./\.\./src/solver/msspec_v2_gpu\.cu|$REPO/src/solver/msspec_v2_gpu.cu|" \
    -e 's|/usr/bin/nvcc -arch=sm_86 -O2 -Xcompiler=-fPIC -c \$< -o \$@|$(NVCC) $(GPU_NVCC_FLAGS) -c $< -o $@|' \
    -e 's|^EXTRA_LINK_FLAGS = -lcudart|EXTRA_LINK_FLAGS = $(MAGMA_LINK_FLAGS) -lcudart|' \
    "$MK"
for pattern in "$REPO/src/solver/msspec_v2_gpu.cu" '$(NVCC) $(GPU_NVCC_FLAGS) -c' '$(MAGMA_LINK_FLAGS) -lcudart'; do
    grep -qF "$pattern" "$MK" || { echo "error: $MK does not contain: $pattern" >&2; exit 1; }
done

echo "[4/5] build phagen and spec Fortran modules (make all, as rebuild.sh)"
make -C "$F_PHAGEN" all PYTHON=python3 NO_VENV=1 BUILDDIR="$F_PHAGEN/build" VERBOSE=1
make -C "$F_SPEC" all PYTHON=python3 NO_VENV=1 BUILDDIR="$F_SPEC/build" VERBOSE=1

echo "[5/5] check the GPU solver symbols in the module"
SO="$F_SPEC/_phd_mi_noso_nosp_nosym.so"
[ -f "$SO" ] || { echo "error: $SO was not built" >&2; exit 1; }
nm -D --defined-only "$SO" | grep -q ' msspec_v2_gpu_solve_$' || { echo "error: msspec_v2_gpu_solve_ missing in $SO" >&2; exit 1; }
echo "built: $OUT"
echo "use: export MSSPEC_GPU_ROOT=$OUT"
echo "     export MSSPEC_GPU_LIB=$SO"
