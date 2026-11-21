"""Gate 1: check that a test MsSpec tree gives the same result as a reference tree.

usage: python3 v2/src/check_identical.py <reference_tree> <test_tree> <output_dir> <case.json> [...]

<reference_tree> and <test_tree> are folders that contain msspec_python3/src
(for example third_party/msspec_standard and v2/msspec_dev).

For every case, src/run/run_case.py runs once with each tree. Each run uses an
absolute tree path in PYTHONPATH, OPENBLAS_NUM_THREADS=20, no taskset and
MPLBACKEND=Agg. Every run has its own scratch folder inside <output_dir>.

For each case it prints the return codes, the SHA-256 of both result.npz files,
and IDENTICAL or DIFFERENT. IDENTICAL means the two result.npz files have the
same SHA-256. For DIFFERENT it also prints, per energy block,
max |I_test - I_reference| / max |I_reference| (I = cross_section).

Exit code 1 if any case is DIFFERENT or FAILED (a run returned non-zero or
did not write result.npz). Exit code 2 if <output_dir> already holds a result
for one of the cases (run_case.py never overwrites a result).
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
RUN_CASE = PROJECT / "src" / "run" / "run_case.py"


def tree_env(tree):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "msspec_python3" / "src")
    env["MPLBACKEND"] = "Agg"
    env["OPENBLAS_NUM_THREADS"] = "20"
    return env


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def msspec_location(label, tree, scratch_root):
    """Print which msspec package file a tree really imports (proof of the tree used)."""
    scratch = scratch_root / f"probe_{label}"
    scratch.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [sys.executable, "-c", "import msspec.calculator as c; print(c.__file__)"],
        cwd=scratch, env=tree_env(tree), capture_output=True, text=True)
    where = probe.stdout.strip() if probe.returncode == 0 else f"IMPORT FAILED rc={probe.returncode}: {probe.stderr.strip()}"
    print(f"{label} tree imports msspec from: {where}")
    return probe.returncode == 0


def run_case(label, tree, case_path, case_dir):
    """Run run_case.py once with one tree. Return (return code, result.npz path)."""
    run_dir = case_dir / label
    scratch = run_dir / "scratch"
    out = run_dir / "out"
    scratch.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(RUN_CASE), str(case_path), str(out)],
        cwd=scratch, env=tree_env(tree), capture_output=True, text=True)
    (run_dir / "run.log").write_text(
        f"return code {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
    return proc.returncode, out / "result.npz"


def energy_blocks(arrays):
    """Return the block indices k of the keys cross_section_<k>, sorted."""
    return sorted(int(key.rsplit("_", 1)[1]) for key in arrays if key.startswith("cross_section_"))


def compare_arrays(ref, test):
    """Print the key list and the per-energy-block relative difference."""
    print(f"  reference keys: {', '.join(f'{k} {ref[k].dtype} {ref[k].shape}' for k in ref)}")
    print(f"  test keys:      {', '.join(f'{k} {test[k].dtype} {test[k].shape}' for k in test)}")
    if sorted(ref) != sorted(test):
        print("  KEY MISMATCH between reference and test")
        return False
    same_bytes = True
    for key in ref:
        if ref[key].dtype != test[key].dtype or ref[key].shape != test[key].shape \
                or ref[key].tobytes() != test[key].tobytes():
            same_bytes = False
    print(f"  every array has the same dtype, shape and bytes: {same_bytes}")
    for k in energy_blocks(ref):
        i_ref = ref[f"cross_section_{k}"]
        i_test = test[f"cross_section_{k}"]
        if i_ref.shape != i_test.shape:
            print(f"  energy block {k}: shape differs {i_ref.shape} vs {i_test.shape}")
            continue
        scale = float(np.max(np.abs(i_ref)))
        diff = float(np.max(np.abs(i_test - i_ref)))
        print(f"  energy block {k}: max |I_test - I_reference| = {diff:.6e}, "
              f"max |I_reference| = {scale:.6e}, ratio = {diff / scale:.6e}")
    return same_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_tree")
    parser.add_argument("test_tree")
    parser.add_argument("output_dir")
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args(argv)

    reference = Path(args.reference_tree).resolve()
    test = Path(args.test_tree).resolve()
    output_dir = Path(args.output_dir).resolve()
    cases = [Path(c).resolve() for c in args.cases]
    for case in cases:
        if not case.is_file():
            parser.error(f"case file not found: {case}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        existing = output_dir / case.stem / "reference" / "out" / "result.npz"
        existing_test = output_dir / case.stem / "test" / "out" / "result.npz"
        if existing.exists() or existing_test.exists():
            print(f"output for {case.stem} already exists under {output_dir}; choose a new output directory")
            return 2

    print(f"reference tree: {reference}")
    print(f"test tree:      {test}")
    print(f"output dir:     {output_dir}")
    if not msspec_location("reference", reference, output_dir) or not msspec_location("test", test, output_dir):
        print("FAILED: msspec cannot be imported from one of the trees")
        return 1

    counts = {"IDENTICAL": 0, "DIFFERENT": 0, "FAILED": 0}
    for case in cases:
        case_dir = output_dir / case.stem
        print(f"\ncase: {case.stem}")
        rc_ref, npz_ref = run_case("reference", reference, case, case_dir)
        rc_test, npz_test = run_case("test", test, case, case_dir)
        print(f"  return codes: reference={rc_ref} test={rc_test}")
        if rc_ref != 0 or rc_test != 0 or not npz_ref.exists() or not npz_test.exists():
            print(f"  FAILED (see {case_dir}/reference/run.log and {case_dir}/test/run.log)")
            counts["FAILED"] += 1
            continue
        sha_ref = sha256_of(npz_ref)
        sha_test = sha256_of(npz_test)
        print(f"  sha256 reference result.npz: {sha_ref}")
        print(f"  sha256 test      result.npz: {sha_test}")
        with np.load(npz_ref) as zr, np.load(npz_test) as zt:
            ref = {k: zr[k] for k in zr.files}
            test_arrays = {k: zt[k] for k in zt.files}
            compare_arrays(ref, test_arrays)
        if sha_ref == sha_test:
            verdict = "IDENTICAL"
        else:
            verdict = "DIFFERENT"
        counts[verdict] += 1
        print(f"  {verdict}")

    total = len(cases)
    print(f"\nsummary: {total} cases, IDENTICAL {counts['IDENTICAL']}, "
          f"DIFFERENT {counts['DIFFERENT']}, FAILED {counts['FAILED']}")
    return 0 if counts["IDENTICAL"] == total else 1


if __name__ == "__main__":
    sys.exit(main())
