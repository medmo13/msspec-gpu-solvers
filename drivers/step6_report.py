"""Step 6 report: per-material tables from results/step6 (read only). usage: python3 -B drivers/step6_report.py"""
import json
from collections import defaultdict
from pathlib import Path

import paths

R = paths.RESULTS_DIR / "step6"


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def manifest_n():
    import csv
    with open(paths.CASES_DIR / "manifest.csv", newline="") as f:
        return {r["id"]: int(r["n"]) for r in csv.DictReader(f)}


def table(name, rows, acc, methods):
    print(f"\n== {name}: {len(set(r['case'] for r in rows))} cases ==")
    nmap = manifest_n()
    by = defaultdict(dict)
    for r in rows:
        by[(r["material"], nmap[r["case"]])][r["method"]] = r
    accby = {(a["case"], a["method"]): a for a in acc}
    for mat in ["copper", "silicon", "tungsten"]:
        print(f"\n[{mat}]  n  stock_s  best(variant, s, speedup)  max|d| per method  failures")
        for (m, n) in sorted(k for k in by if k[0] == mat):
            d = by[(m, n)]
            st = d.get("stock")
            s_stock = st["seconds"] if st and st["status"] == "ok" else None
            gpu_ok = [(x["seconds"], k) for k, x in d.items() if k != "stock" and x["status"] == "ok"]
            best = min(gpu_ok) if gpu_ok else None
            spd = f"{s_stock / best[0]:.2f}x" if (best and s_stock) else "-"
            bstr = f"{best[1]} {best[0]:.2f} {spd}" if best else "none"
            dd = []
            fails = []
            for k in methods:
                if k == "stock":
                    continue
                x = d.get(k)
                if x is None:
                    dd.append(f"{k}=missing")
                    continue
                if x["status"] != "ok":
                    fails.append(f"{k}:{x['status']}/rc{x['rc']}")
                    dd.append(f"{k}=-")
                    continue
                a = accby.get((x["case"], k))
                mx = a["max_d"] if a else None
                dd.append(f"{k}={mx:.2e}" if mx is not None else f"{k}=?")
            sstr = f"{s_stock:.2f}" if s_stock else "-"
            print(f"  {n:6d}  {sstr:>8}  {bstr}")
            print(f"          d: {'; '.join(dd)}")
            print(f"          failures: {', '.join(fails) if fails else 'none'}")


def main():
    polar_acc = jl(R / "polar_accuracy.jsonl")
    stereo_acc = jl(R / "stereo_accuracy.jsonl")
    all_polar = jl(R / "all_polar.jsonl")
    # all_polar: 5 reused copper cases + 25 new; accuracy for the new rows comes from polar_accuracy.jsonl
    acc_all = polar_acc + [dict(case=r["case"], method=r["method"], max_d=r["max_d"]) for r in all_polar
                           if r["source"] != "step6/polar_runs.jsonl" and r["method"] != "stock"]
    pol_methods = ["stock", "cusolver_refined", "cusolver_unrefined", "magma_refined", "magma_unrefined",
                   "magma_host_refined", "magma_host_unrefined", "own_refined", "own_unrefined"]
    rows = [dict(r, method=r["method"], case=r["case"]) for r in all_polar]
    table("POLAR (5 reused copper + 25 new)", rows, acc_all, pol_methods)
    stereo_runs = jl(R / "stereo_runs.jsonl")
    stereo_timed = [r for r in stereo_runs if r["kind"] == "timed"]
    st_rows = [dict(case=r["case"], material=r["material"], method=r["method"], status=r["status"],
                    rc=r["rc"], seconds=r["seconds"]) for r in stereo_timed]
    table("STEREO (25 new; the 5 copper cases of step 5 are not in this table)", st_rows, stereo_acc,
          ["stock", "own_refined", "own_unrefined"])
    print("\ncounts:")
    print(" polar_runs rows", len(jl(R / "polar_runs.jsonl")), "timed", sum(r["kind"] == "timed" for r in jl(R / "polar_runs.jsonl")))
    print(" polar_accuracy rows", len(polar_acc), "with max_d", sum(a["max_d"] is not None for a in polar_acc),
          "fail", sum(a["pass"] is False for a in polar_acc))
    print(" stereo_runs rows", len(stereo_runs), "timed", len(stereo_timed))
    print(" stereo_accuracy rows", len(stereo_acc), "with max_d", sum(a["max_d"] is not None for a in stereo_acc),
          "fail", sum(a["pass"] is False for a in stereo_acc))
    print(" all_polar rows", len(all_polar), "sources", sorted(set(r["source"] for r in all_polar)))


if __name__ == "__main__":
    main()
