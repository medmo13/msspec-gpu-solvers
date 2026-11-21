"""magma_host merge: step 2 timed rows (runs_final.jsonl) + the new magma_host timed rows, written to the staging
folder, then the plot of src/plot_step2.py with METHODS extended to 8 (its list is hardcoded to 6; the file is
not edited). usage: python3 drivers/magma_host_merge.py <out_dir>"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

import paths  # noqa: E402

V2 = paths.REPO
sys.path.insert(0, str(V2 / "src"))
import plot_step2 as ps  # noqa: E402


def timed(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows, [r for r in rows if r["kind"] == "timed"]


def main():
    out = Path(sys.argv[1]).resolve()
    step2 = paths.RESULTS_DIR / "step2" / "runs_final.jsonl"
    if not step2.is_file():
        raise SystemExit(f"error: {step2} not found. It is the timed-rows file of the step 2 run "
                         f"(src/run_gpu_variants.py); place it there first")
    old_rows, old_timed = timed(step2)
    new_rows, new_timed = timed(out / "runs.jsonl")
    print("step2 runs_final.jsonl: rows", len(old_rows), "timed", len(old_timed))
    print("magma_host runs.jsonl: rows", len(new_rows), "timed", len(new_timed))
    merged = old_timed + new_timed
    print("merged timed rows:", len(merged))
    if len(merged) != 40:
        raise SystemExit("expected 40 timed rows")
    dst = out / "runs_final8.jsonl"
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in merged))
    print("wrote", dst)
    ps.METHODS = ["cusolver_refined", "cusolver_unrefined", "magma_refined", "magma_unrefined",
                  "own_refined", "own_unrefined", "magma_host_refined", "magma_host_unrefined"]
    sys.argv = ["plot_step2.py", str(dst), str(out / "step2_final8.png")]
    ps.main()


if __name__ == "__main__":
    main()
