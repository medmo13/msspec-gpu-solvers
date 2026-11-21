"""magma_host full run: the step 2 runner (src/run_gpu_variants.py, unchanged) with METHODS set to the two
magma_host methods and WAKEUP_METHOD magma_host_refined. Same clock, same protocol, same seed.
usage: TMPDIR=$REPO/.work python3 drivers/magma_host_full.py <out_dir>"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import paths  # noqa: E402

V2 = paths.REPO
sys.path.insert(0, str(V2 / "src"))
import run_gpu_variants as rg  # noqa: E402

rg.METHODS = ["magma_host_refined", "magma_host_unrefined"]
rg.WAKEUP_METHOD = "magma_host_refined"
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    rg.main()
