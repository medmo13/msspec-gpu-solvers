"""Phase profile of one MsSpec PED (inversion) run, split into phases.

usage: python3 v2/src/profile_run.py <tree> <case.json> <output_dir>

<tree> is a folder that contains msspec_python3/src (for example v2/msspec_dev).
The parent process starts one child process (same file with --child). The child
runs the case the way src/run/run_case.py does, with MSSPEC_V2_PHASE_LOG set so the
Fortran phase timers write fortran_phases.txt. The child times Python phases with
time.perf_counter and wraps PHAGEN and the SPEC Fortran call from outside
calculator.py (no edits to calculator.py). The whole run is also wrapped in cProfile.

Process start: the parent stores time.perf_counter() just before it starts the child
and passes it in the environment variable PROFILE_T_LAUNCH. On Linux perf_counter is
CLOCK_MONOTONIC, a system-wide clock, so the child can subtract it. The parent measures
the whole process as the time from that stamp until subprocess.run returns.

Writes <output_dir>/profile.json (all phases), fortran_phases.txt, cprofile.prof,
cprofile_top25.txt, result.npz and meta.json. OPENBLAS_NUM_THREADS=20, no taskset.
"""
import cProfile
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PHASE_NAMES = ["fortran_matrix_build", "fortran_solve", "fortran_angle_loop",
               "fortran_output_write", "fortran_tau_copyout"]


def parent(tree, case_path, out):
    if (out / "result.npz").exists():
        raise FileExistsError(f"{out / 'result.npz'} exists; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "msspec_python3" / "src")
    env["MPLBACKEND"] = "Agg"
    env["OPENBLAS_NUM_THREADS"] = "20"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MSSPEC_V2_PHASE_LOG"] = str(out / "fortran_phases.txt")
    t_launch = time.perf_counter()
    env["PROFILE_T_LAUNCH"] = repr(t_launch)
    rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child",
                         str(tree), str(case_path), str(out)], env=env).returncode
    t_end = time.perf_counter()
    if rc != 0:
        raise SystemExit(f"child failed with return code {rc}")
    child = json.loads((out / "child_stamps.json").read_text())
    whole = t_end - t_launch
    named = dict(child["phases"])
    named["exit_and_parent"] = t_end - child["t_child_last"]
    phases_sum = sum(named.values())
    profile = dict(
        case_id=child["case_id"], case_path=str(case_path), tree=str(tree),
        msspec_file=child["msspec_file"], matrix_order=child["matrix_order"],
        matrix_order_probe=child["matrix_order_probe"], scan_type=child["scan_type"],
        number_of_angles=child["number_of_angles"], atoms=child["atoms"],
        whole_run_s=whole, phases_s=named, phases_sum_s=phases_sum,
        remainder_s=whole - phases_sum,
        remainder_pct=100.0 * (whole - phases_sum) / whole,
        fortran_phase_calls=child["fortran_calls"],
        python_clock_info=child["clock_info"], python_version=platform.python_version(),
        calls=child["calls"], single_run_no_warmup=True, cprofile_overhead_included=True,
    )
    (out / "profile.json").write_text(json.dumps(profile, indent=1) + "\n")
    print(f"case {child['case_id']} n {child['matrix_order']} whole {whole:.3f} s "
          f"remainder {whole - phases_sum:+.4f} s ({profile['remainder_pct']:+.2f} %)")
    print((out / "cprofile_top25.txt").read_text())


def child(tree, case_path, out):
    t_child = time.perf_counter()
    t_launch = float(os.environ["PROFILE_T_LAUNCH"])
    phases = {"startup": t_child - t_launch}
    t0 = time.perf_counter()
    import numpy as np
    import msspec
    from msspec.calculator import MSSPEC
    import msspec.calculator as calc_mod
    sys.path.insert(0, str(PROJECT / "src" / "cases"))
    from cluster import build_cluster
    phases["import_msspec"] = time.perf_counter() - t0
    # cProfile starts after the imports, so import time is not inflated by the profiler.
    prof = cProfile.Profile()
    prof.enable()

    acc = {k: 0.0 for k in ("phagen", "spec_fortran", "tmatrix_total", "run_spec_total", "scan_total")}
    calls = {k: 0 for k in acc}

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                acc[name] += time.perf_counter() - t
                calls[name] += 1
        return wrapper

    calc_mod.do_phagen = timed("phagen", calc_mod.do_phagen)

    class SpecProxy:
        pass

    proxy = SpecProxy()
    proxy.run = timed("spec_fortran", calc_mod._phd_mi_noso_nosp_nosym.run)
    calc_mod._phd_mi_noso_nosp_nosym = proxy
    # MSSPEC() is a factory; get_tmatrix and run_spec live on _MSCALCULATOR, _get_scan on _PED.
    calc_mod._MSCALCULATOR.get_tmatrix = timed("tmatrix_total", calc_mod._MSCALCULATOR.get_tmatrix)
    calc_mod._MSCALCULATOR.run_spec = timed("run_spec_total", calc_mod._MSCALCULATOR.run_spec)
    calc_mod._PED._get_scan = timed("scan_total", calc_mod._PED._get_scan)

    case = json.loads(Path(case_path).read_text())
    out = Path(out)
    scan = case["scan"]
    energies = case.get("energies_eV", [case["energy_eV"]])

    t0 = time.perf_counter()
    cluster = build_cluster(case)
    calc = MSSPEC(spectroscopy="PED", algorithm="inversion", folder=str(out / "calc"), txt=None)
    calc.set_atoms(cluster)
    phases["setup"] = time.perf_counter() - t0

    def matrix_order():
        return int(max(sum(len(calc.phagenio.tl[int(a.get("proto_index")) - 1][ie]) ** 2 for a in calc.atoms)
                       for ie in range(calc.phagenio.ne)))

    blocks, orders, extract = {}, [], 0.0
    try:
        for k, energy in enumerate(energies):
            theta = np.linspace(*scan["theta_deg"])
            if scan["type"] == "theta_phi":
                data = calc.get_theta_phi_scan(level=case["level"], kinetic_energy=energy, theta=theta,
                                               phi=np.linspace(*scan["phi_deg"]))
            else:
                data = calc.get_theta_scan(level=case["level"], kinetic_energy=energy, theta=theta,
                                           phi=scan["phi_deg"])
            t0 = time.perf_counter()
            last = data.get_last_dset()
            for key in ("theta", "phi", "cross_section"):
                blocks[f"{key}_{k}"] = np.asarray(last[key])
            orders.append(matrix_order())
            extract += time.perf_counter() - t0
    finally:
        t0 = time.perf_counter()
        calc.shutdown()
        phases["shutdown"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not all(np.isfinite(v).all() for v in blocks.values()):
        raise RuntimeError("non-finite result; no points are removed")
    np.savez_compressed(out / "result.npz", **blocks)
    (out / "meta.json").write_text(json.dumps(dict(
        case=case["id"], atoms=len(cluster), energies_eV=energies, matrix_order=orders,
        matrix_order_probe=case.get("matrix_order_probe")), indent=1) + "\n")
    phases["npz_write"] = time.perf_counter() - t0

    # Python-side phases derived from the nested wrappers.
    phases["phagen"] = acc["phagen"]
    phases["tmatrix_python_io"] = acc["tmatrix_total"] - acc["phagen"]
    phases["spec_python_io"] = acc["run_spec_total"] - acc["spec_fortran"]
    phases["result_reading"] = (acc["scan_total"] - acc["tmatrix_total"] - acc["run_spec_total"]) + extract

    # Fortran phases, read back from the file written by V2PT_WRITE.
    fort = {}
    for line in (out / "fortran_phases.txt").read_text().splitlines():
        name, secs, ncalls = line.split()
        fort[name] = (float(secs), int(ncalls))
    fsum = {n: fort[n][0] for n in PHASE_NAMES}
    fsum["fortran_angle_loop"] = fort["fortran_angle_loop"][0] - fort["fortran_output_write"][0]
    for n in PHASE_NAMES:
        phases[n] = fsum[n]
    phases["fortran_other"] = acc["spec_fortran"] - sum(fort[n][0] for n in PHASE_NAMES)

    prof.disable()
    t0 = time.perf_counter()
    prof.dump_stats(str(out / "cprofile.prof"))
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(25)
    (out / "cprofile_top25.txt").write_text(buf.getvalue())
    phases["cprofile_dump"] = time.perf_counter() - t0

    # Keep the phase keys in a fixed order (ASCII for the table).
    order = ["startup", "import_msspec", "setup", "phagen", "tmatrix_python_io",
             "spec_python_io", "fortran_matrix_build", "fortran_solve", "fortran_angle_loop",
             "fortran_output_write", "fortran_tau_copyout", "fortran_other",
             "result_reading", "shutdown", "npz_write", "cprofile_dump"]
    phases = {k: phases[k] for k in order}
    calls_out = dict(calls)
    calls_out.update({n: fort[n][1] for n in PHASE_NAMES})

    stamp = dict(
        case_id=case["id"], msspec_file=msspec.__file__, matrix_order=orders[0],
        matrix_order_probe=case.get("matrix_order_probe"), scan_type=scan["type"],
        number_of_angles=int(np.linspace(*scan["theta_deg"]).size *
                             (np.linspace(*scan["phi_deg"]).size if scan["type"] == "theta_phi" else 1)),
        atoms=len(cluster), phases=phases, calls=calls_out, fortran_calls=fort,
        clock_info=str(time.get_clock_info("perf_counter")), t_child_last=time.perf_counter(),
    )
    (out / "child_stamps.json").write_text(json.dumps(stamp, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:
        parent(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    else:
        raise SystemExit(__doc__)
