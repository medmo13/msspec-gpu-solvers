"""Accuracy check of one result.npz against a reference result.npz.

usage: python3 src/check_accuracy.py <result.npz> <reference result.npz>

Prints the keys of both files, then for every energy block k (cross_section_<k>)
the relative difference d = (I - I_ref) / max|I_ref|, where I = cross_section.
Prints max|d| over all blocks and PASS if max|d| <= 0.01, otherwise FAIL.
Exit code 0 on PASS, 1 on FAIL, 2 on a key or shape mismatch.
"""
import sys

import numpy as np

TOLERANCE = 0.01


def energy_blocks(arrays):
    return sorted(int(key.rsplit("_", 1)[1]) for key in arrays if key.startswith("cross_section_"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: check_accuracy.py <result.npz> <reference result.npz>")
        return 2
    test_path, ref_path = argv
    with np.load(test_path) as zt, np.load(ref_path) as zr:
        test = {k: zt[k] for k in zt.files}
        ref = {k: zr[k] for k in zr.files}
    print(f"test npz keys: {', '.join(f'{k} {v.dtype} {v.shape}' for k, v in test.items())}")
    print(f"ref  npz keys: {', '.join(f'{k} {v.dtype} {v.shape}' for k, v in ref.items())}")
    if sorted(test) != sorted(ref):
        print("KEY MISMATCH between test and reference")
        return 2
    blocks = energy_blocks(ref)
    if blocks != energy_blocks(test) or not blocks:
        print("energy block mismatch between test and reference")
        return 2
    worst = 0.0
    for k in blocks:
        i_t = test[f"cross_section_{k}"]
        i_r = ref[f"cross_section_{k}"]
        if i_t.shape != i_r.shape:
            print(f"energy block {k}: shape differs {i_t.shape} vs {i_r.shape}")
            return 2
        scale = float(np.max(np.abs(i_r)))
        d = float(np.max(np.abs((i_t - i_r) / scale)))
        print(f"energy block {k}: max|d| = {d:.6e}  (max|I_ref| = {scale:.6e})")
        worst = max(worst, d)
    print(f"max|d| over all blocks = {worst:.6e}, tolerance {TOLERANCE}")
    if worst <= TOLERANCE:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
