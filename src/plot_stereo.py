"""Step 5 plot: stereographic maps of stock, own_refined, own_unrefined per case, house style.

usage: python3 -B v2/src/plot_stereo.py <step5_dir>
Reads <step5_dir>/runs.jsonl (timed rows with status ok) and writes <step5_dir>/<case>_stereo.png.
Style follows research_symmetric_inversion/ped_stereo/methods_scan.py (_plot_stereo): polar axes, pcolormesh
shading='gouraud' on a regular (phi, r) mesh, r = tan(theta/2), theta zero 'N', direction -1, radial ticks at
0/30/60/90 deg, cmap 'gray' for the cross sections with one shared vmin/vmax per case, 'magma' for
|I - I_stock| / max|I_stock|, one colorbar per panel.
Each ring (one theta) has its own phi points, so each ring is interpolated (periodic in phi) onto a common
phi axis 0..360 (181 points, 360 included so the gouraud seam closes). A ring with one point is a constant.
Top row: stock, own_refined, own_unrefined. Bottom row: |d| of own_refined, |d| of own_unrefined.
The max|d| in the titles is computed here from the raw points, and must equal accuracy.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
CASE_DIR = V2 / "cases" / "stereographic"
PHI_AXIS = np.linspace(0.0, 360.0, 181)
METHODS = ["stock", "own_refined", "own_unrefined"]


def load_rows(step_dir):
    rows = [json.loads(l) for l in (step_dir / "runs.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        if r["kind"] == "timed" and r["status"] == "ok":
            if (r["case"], r["method"]) in out:
                raise SystemExit(f"two ok timed rows for {r['case']} {r['method']}")
            out[(r["case"], r["method"])] = r
    return out


def ring_grid(th, ph, cs):
    """Rows = theta rings (ascending), columns = PHI_AXIS. Periodic interpolation per ring."""
    thetas = np.unique(th)
    grid = np.empty((len(thetas), len(PHI_AXIS)))
    for j, t in enumerate(thetas):
        m = th == t
        p, c = ph[m], cs[m]
        o = np.argsort(p)
        p, c = p[o], c[o]
        xp = np.concatenate([p - 360.0, p, p + 360.0])
        fp = np.tile(c, 3)
        if not np.all(np.diff(xp) > 0):
            raise SystemExit(f"theta ring {t}: phi values not strictly increasing")
        grid[j] = np.interp(PHI_AXIS, xp, fp)
    if not np.all(np.isfinite(grid)):
        raise SystemExit("NaN or inf cells in the interpolated grid")
    return thetas, grid


def load_result(path):
    with np.load(path) as z:
        print("npz keys", path, z.files, flush=True)
        d = {k: z[k] for k in z.files}
    blocks = sorted(int(k.rsplit("_", 1)[1]) for k in d if k.startswith("cross_section_"))
    if blocks != [0]:
        raise SystemExit(f"{path}: expected one energy block 0, got {blocks}")
    return d["theta_0"], d["phi_0"], d["cross_section_0"]


def plot_case(step_dir, case_id, rows):
    data = {}
    for m in METHODS:
        if (case_id, m) not in rows:
            raise SystemExit(f"{case_id}: no ok timed row for {m}; no plot")
        data[m] = load_result(Path(rows[(case_id, m)]["run_dir"]) / "result.npz")
    th0, ph0, _ = data["stock"]
    for m in METHODS[1:]:
        if not (np.array_equal(data[m][0], th0) and np.array_equal(data[m][1], ph0)):
            raise SystemExit(f"{case_id}: theta/phi of {m} differ from stock")
    ref = data["stock"][2]
    scale = float(np.max(np.abs(ref)))
    dmax = {}
    for m in METHODS[1:]:
        dmax[m] = float(np.max(np.abs((data[m][2] - ref) / scale)))
    grids, thetas = {}, None
    for m in METHODS:
        thetas, grids[m] = ring_grid(data[m][0], data[m][1], data[m][2])
    diff = {m: np.abs(grids[m] - grids["stock"]) / scale for m in METHODS[1:]}
    vmin = min(float(g.min()) for g in grids.values())
    vmax = max(float(g.max()) for g in grids.values())
    dvmax = max(float(v.max()) for v in diff.values())
    R = np.tan(np.radians(thetas / 2.0))
    PHI_mesh, R_mesh = np.meshgrid(np.radians(PHI_AXIS), R)
    fig = Figure(figsize=(16, 10))
    FigureCanvasAgg(fig)
    panels = [("stock", grids["stock"], "stock (this session, %.1f s)" % rows[(case_id, "stock")]["seconds"]),
              ("own_refined", grids["own_refined"], "own_refined (%.1f s)" % rows[(case_id, "own_refined")]["seconds"]),
              ("own_unrefined", grids["own_unrefined"],
               "own_unrefined (%.1f s)" % rows[(case_id, "own_unrefined")]["seconds"])]
    for i, (name, g, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1, projection="polar")
        im = ax.pcolormesh(PHI_mesh, R_mesh, g, shading="gouraud", cmap="gray", vmin=vmin, vmax=vmax)
        _style(ax, title)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.1)
    for i, m in enumerate(METHODS[1:]):
        ax = fig.add_subplot(2, 3, i + 4, projection="polar")
        im = ax.pcolormesh(PHI_mesh, R_mesh, diff[m], shading="gouraud", cmap="magma", vmin=0.0, vmax=dvmax)
        _style(ax, f"{m}: |d| = |I - I_stock| / max|I_stock|\nmax|d| = {dmax[m]:.3e} (raw points)")
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.1)
    case = json.loads((CASE_DIR / f"{case_id}.json").read_text())
    n = rows[(case_id, "stock")]["n"]
    fig.suptitle(f"Stereographic PED, copper(111), {case['level']}, E = {case['energy_eV']} eV, "
                 f"radius {case['radius_A']} A, n = {n}", fontsize=14)
    out = step_dir / f"{case_id}_stereo.png"
    FigureCanvasAgg(fig).print_figure(str(out), dpi=200, bbox_inches="tight")
    print(out, "max|d|", dmax, flush=True)
    return out, dmax


def _style(ax, title):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, np.tan(np.radians(45.0)))  # data stop at 80 deg; the 90 deg ring is still drawn
    ax.set_yticks(np.tan(np.radians(np.arange(0, 91, 30) / 2.0)))
    ax.set_yticklabels([f"{t}°" for t in range(0, 91, 30)])
    ax.set_title(title, fontsize=10, pad=12)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    step_dir = Path(sys.argv[1]).resolve()
    rows = load_rows(step_dir)
    acc = {(r["case"], r["method"]): r for r in map(json.loads, (step_dir / "accuracy.jsonl").read_text().splitlines())}
    cases = sorted({c for c, _ in rows})
    for case_id in cases:
        _, dmax = plot_case(step_dir, case_id, rows)
        for m in METHODS[1:]:
            a = acc.get((case_id, m))
            # accuracy.jsonl holds check_accuracy.py's 7-significant-digit print, so compare at that precision
            if a is None or a["max_d"] is None or abs(a["max_d"] - dmax[m]) > 1e-6 * max(1.0, a["max_d"]):
                raise SystemExit(f"{case_id} {m}: plot max|d| {dmax[m]} differs from accuracy.jsonl {a}")


if __name__ == "__main__":
    main()
