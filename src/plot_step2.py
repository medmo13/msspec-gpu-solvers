"""Plot and table for step 2 (one timed run per GPU variant per copper polar case).

usage: python3 src/plot_step2.py <runs.jsonl> <step2.png>
Prints the keys of the first row, then the table from the timed rows (wake-up rows are skipped),
then writes the figure. Left panel: whole-process seconds vs matrix order n (log-log), stock as black line.
Right panel: max|d| vs n (log y), horizontal line at the 0.01 pass limit.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ["cusolver_refined", "cusolver_unrefined", "magma_refined", "magma_unrefined", "own_refined", "own_unrefined"]
TOL = 0.01


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    runs_path, png = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    print("keys of first row:", ", ".join(sorted(rows[0])))
    print("rows in file:", len(rows), " wake-up rows:", sum(r["kind"] == "wakeup" for r in rows),
          " timed rows:", sum(r["kind"] == "timed" for r in rows))
    timed = [r for r in rows if r["kind"] == "timed"]
    cases = sorted({(r["n"], r["case"], r["stock_seconds"]) for r in timed})

    print()
    header = f"{'case':<62} {'n':>6} {'stock_s':>9}  {'method':<18} {'seconds':>10} {'speedup':>8} {'max_d':>11} {'status':<8}"
    print(header)
    print("-" * len(header))
    for n, case, stock_s in cases:
        for m in METHODS:
            r = [x for x in timed if x["case"] == case and x["method"] == m][0]
            sec = f"{r['seconds']:.3f}"
            spd = f"{r['speedup']:.3f}" if r["speedup"] is not None else "-"
            md = f"{r['max_d']:.3e}" if r["max_d"] is not None else "-"
            status = r["status"] if r["status"] == "ok" else f"{r['status']} rc={r['rc']}"
            print(f"{case:<62} {n:>6} {stock_s:>9.3f}  {m:<18} {sec:>10} {spd:>8} {md:>11} {status:<8}")

    fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(13, 5.5))
    ns = [c[0] for c in cases]
    ax_t.plot(ns, [c[2] for c in cases], color="black", marker="o", lw=2, label="stock CPU (reference)")
    colors = plt.cm.tab10.colors
    all_secs = [c[2] for c in cases] + [r["seconds"] for r in timed]
    y_top = max(all_secs) * 2.5
    for i, m in enumerate(METHODS):
        mr = sorted([r for r in timed if r["method"] == m], key=lambda r: r["n"])
        ok = [r for r in mr if r["status"] == "ok"]
        ax_t.plot([r["n"] for r in ok], [r["seconds"] for r in ok], marker="s", color=colors[i], label=m)
        bad = [r for r in mr if r["status"] != "ok"]
        for r in bad:
            ax_t.plot([r["n"]], [y_top], marker="x", color=colors[i], ms=10, mew=2)
            label = f"{m}: timeout" if r["status"] == "timeout" else f"{m}: failed rc={r['rc']}"
            ax_t.annotate(label, (r["n"], y_top), textcoords="offset points", xytext=(-8, -2 - 11 * i),
                          fontsize=7, color=colors[i], ha="right")
        dm = [r for r in ok if r["max_d"] is not None and r["max_d"] > 0]
        ax_d.plot([r["n"] for r in dm], [r["max_d"] for r in dm], marker="s", color=colors[i], label=m)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_ylim(top=y_top * 1.6)
    ax_t.set_xlabel("matrix order n")
    ax_t.set_ylabel("whole-process time (s)")
    ax_t.set_title("Whole-process time, copper 111 polar scans")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8, loc="upper left")
    ax_d.axhline(TOL, color="red", ls="--", label="pass limit 0.01")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("matrix order n")
    ax_d.set_ylabel("max|d| against stock result")
    ax_d.set_title("Accuracy against stock reference (ok runs)")
    ax_d.grid(True, which="both", alpha=0.3)
    ax_d.legend(fontsize=8)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print("wrote", png)


if __name__ == "__main__":
    main()
