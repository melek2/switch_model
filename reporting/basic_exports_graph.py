# Copyright (c) 2025 Melek Ben-Ayed
# Licensed under the Apache License, Version 2.0

"""
Reporting module for graph_test:
- Writes summary CSVs into <outputs>/Summaries
- Writes PNG plots into <outputs>/Plots
Flags:
  --export-all                  export all CSVs and plots
  --export-capacities           CSV: capacity by tech x period
  --export-tech-dispatch        CSV: dispatch by tech x period
  --export-transmission         CSV: transmission summaries
  --plot-capacities             PNG: stacked capacity by tech x period
  --plot-tech-dispatch          PNG: stacked dispatch by tech x period
"""

import os
import math
import sys, subprocess, shlex

# Use a non-interactive backend so we can save PNGs in headless runs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import switch_model.reporting as export
import os, sys, contextlib, argparse
import importlib.util


def _listed_in_modules(mod, dotted: str) -> bool:
    # Honors either modules.txt entries or --include-modules; falls back to “present in site-packages”
    names = []
    # Switch stores extra modules in options.include_modules (list) if passed
    try:
        inc = getattr(mod.options, "include_modules", None)
        if inc:
            names.extend(inc if isinstance(inc, (list, tuple)) else [inc])
    except Exception:
        pass
    # Try to read modules.txt path if available (optional)
    try:
        mlist = getattr(mod.options, "module_list", None)
        if mlist and os.path.isfile(mlist):
            with open(mlist, "r") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        names.append(s)
    except Exception:
        pass
    # If explicitly listed, we’re good
    if any(n.strip() == dotted for n in names):
        return True
    # Otherwise, only run if the package actually exists AND user didn’t ask for it explicitly?
    # Keep it strict per your request: require explicit listing.
    return False


@contextlib.contextmanager
def _pushd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _auto_run_registered_graphs(mod, outdir: str):
    """
    Run all @graph-registered plots using the Switch graph engine, in-process,
    only if 'switch_model.tools.graph' is listed in modules.txt or via --include-modules.
    """
    if not _listed_in_modules(mod, "switch_model.tools.graph"):
        return  # user didn’t request graphs via modules.txt; do nothing

    # Make sure the graph engine exists
    if importlib.util.find_spec("switch_model.tools.graph.cli") is None:
        sys.stderr.write(
            "[graphs] switch_model.tools.graph is listed but not importable; skipping.\n"
        )
        return

    # Import the minimal public API used by cli_graph.py
    from switch_model.tools.graph.main import Scenario
    from switch_model.tools.graph.cli import add_arguments, graph_scenarios_from_cli

    # Build args with the same defaults the CLI would use
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args = parser.parse_args([])  # parse with all defaults

    # Match cli_graph.py default graph_dir if None; use “Graphs” for clarity
    if getattr(args, "graph_dir", None) in (None, ""):
        setattr(args, "graph_dir", "Graphs")

    # Ensure PNG output (if the arg exists in this version)
    if hasattr(args, "format"):
        setattr(args, "format", "png")
    if hasattr(args, "graph_format"):
        setattr(args, "graph_format", "png")

    # If the CLI supports selecting graphs, default to "all"
    if hasattr(args, "graphs") and (args.graphs in (None, "", [])):
        try:
            setattr(args, "graphs", ["all"])  # some versions want list
        except Exception:
            setattr(args, "graphs", "all")

    # The engine expects paths relative to CWD; run from outdir
    with _pushd(outdir):
        # Single scenario rooted at outputs directory, like cli_graph
        scenarios = [Scenario(rel_path=".", name="")]
        try:
            graph_scenarios_from_cli(scenarios=scenarios, args=args)
        except Exception as e:
            sys.stderr.write(f"[graphs] Warning: graphs engine raised: {e}\n")


def _val(x):
    """Safely get a numeric value from a Pyomo Var/Param/Expression; 0.0 if undefined."""
    try:
        return float(pyo.value(x))
    except Exception:
        return 0.0


def define_arguments(argparser):
    # CSV flags
    argparser.add_argument("--export-all", action="store_true", default=False)
    argparser.add_argument("--export-capacities", action="store_true", default=False)
    argparser.add_argument("--export-tech-dispatch", action="store_true", default=False)
    argparser.add_argument("--export-transmission", action="store_true", default=False)
    # Plot flags
    argparser.add_argument("--plot-capacities", action="store_true", default=False)
    argparser.add_argument("--plot-tech-dispatch", action="store_true", default=False)

    argparser.add_argument(
        "--auto-graphs",
        action="store_true",
        default=False,
        help="After solve, run all registered @graph functions over outputs.",
    )
    argparser.add_argument(
        "--graphs",
        default="all",
        help="Comma-separated list of graph names to run (default: all).",
    )
    argparser.add_argument(
        "--graph-format",
        default="png",
        help="Output format for graphs (png, svg, html, etc.; default: png).",
    )


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _capacity_table(mod):
    """
    Return (period_labels, techs, matrix[tech][period]) for installed capacity.
    Uses pyo.value(...) to unwrap Pyomo expressions safely and guards missing indices.
    """
    techs = list(mod.GENERATION_TECHNOLOGIES)
    periods = list(mod.PERIODS)

    # Some Switch builds define a (g, p) index set for capacity. Try to discover it.
    cap = []
    for gt in techs:
        row = []
        for p in periods:
            total = 0.0
            for g in mod.GENERATION_PROJECTS:
                if mod.gen_tech[g] != gt:
                    continue
                # Only add if that index exists; otherwise treat as zero.
                if hasattr(mod, "GenCapacity"):
                    try:
                        if (g, p) in mod.GenCapacity:
                            total += _val(mod.GenCapacity[g, p])
                        else:
                            # Some formulations allow capacity only in certain periods;
                            # zero if not defined.
                            total += 0.0
                    except Exception:
                        total += 0.0
            row.append(total)
        cap.append(row)
    return periods, techs, cap


def _dispatch_table(mod):
    """
    Return (period_labels, techs, matrix[tech][period]) for total dispatch.
    Uses pyo.value(...) to unwrap Pyomo Vars and guards index membership.
    """
    techs = list(mod.GENERATION_TECHNOLOGIES)
    periods = list(mod.PERIODS)
    has_pdp = hasattr(mod, "PROJ_DISPATCH_POINTS")

    disp = []
    for gt in techs:
        row = []
        for p in periods:
            total = 0.0
            for g in mod.GENERATION_PROJECTS:
                if mod.gen_tech[g] != gt:
                    continue
                for t in mod.TPS_IN_PERIOD[p]:
                    if has_pdp and (g, t) not in mod.PROJ_DISPATCH_POINTS:
                        continue
                    try:
                        total += _val(mod.DispatchGen[g, t])
                    except Exception:
                        # If not defined or infeasible index, treat as zero
                        total += 0.0
            row.append(total)
        disp.append(row)
    return periods, techs, disp


def _stacked_bar(periods, techs, matrix, title, ylabel, outfile):
    """
    matrix is list of rows per tech, each row is values per period.
    """
    _ensure_dir(os.path.dirname(outfile))
    x = range(len(periods))
    # transpose matrix to period-major columns for stacking
    # but matplotlib only needs cumulative bottoms; we compute iteratively
    bottoms = [0.0] * len(periods)

    plt.figure(figsize=(12, 7))
    for i, gt in enumerate(techs):
        vals = [matrix[i][j] for j in range(len(periods))]
        plt.bar(x, vals, bottom=bottoms, label=str(gt))
        bottoms = [bottoms[j] + vals[j] for j in range(len(periods))]

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(list(x), [str(p) for p in periods], rotation=0)
    # modest legend, move outside if many techs
    if len(techs) <= 12:
        plt.legend(loc="best", fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def _run_graphs_cli(outdir: str, graphs: str, fmt: str) -> int:
    """
    Invoke the graphs CLI on the outputs dir. Returns process returncode.
    We try the known entry modules in order.
    """
    py = shlex.quote(sys.executable)
    od = shlex.quote(outdir)
    gs = shlex.quote(graphs)
    fm = shlex.quote(fmt)

    candidates = [
        "switch_model.tools.graph.cli_graph",
        "switch_model.tools.graph.main",
    ]
    last_rc = 1
    for mod in candidates:
        cmd = f'{py} -m {mod} --outputs-dir "{od}" --graphs {gs} --format {fm}'
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # Surface useful logs if something goes wrong
            if proc.returncode != 0:
                sys.stderr.write(
                    f"[graphs] Tried {mod} -> nonzero return code {proc.returncode}\n"
                )
                sys.stderr.write(proc.stderr or "")
            else:
                # Optional: write stdout to Switch log stream
                sys.stdout.write(proc.stdout or "")
                return 0
            last_rc = proc.returncode
        except Exception as e:
            sys.stderr.write(f"[graphs] Exception invoking {mod}: {e}\n")
            last_rc = 1
    return last_rc


def post_solve(mod, outdir):
    # If --export-all, enable everything
    if getattr(mod.options, "export_all", False):
        mod.options.export_capacities = True
        mod.options.export_tech_dispatch = True
        mod.options.export_transmission = True
        mod.options.plot_capacities = True
        mod.options.plot_tech_dispatch = True

    summaries_dir = os.path.join(outdir, "Summaries")
    plots_dir = os.path.join(outdir, "Plots")
    _ensure_dir(summaries_dir)
    _ensure_dir(plots_dir)

    # ===== CSV: capacity =====
    if getattr(mod.options, "export_capacities", False):
        periods, techs, cap = _capacity_table(mod)
        export.write_table(
            mod,
            techs,
            output_file=os.path.join(summaries_dir, "capacity_by_tech_periods.csv"),
            headings=("gentech",) + tuple(periods),
            values=lambda m, gt: (gt,)
            + tuple(cap[techs.index(gt)][j] for j in range(len(periods))),
        )

    # ===== CSV: dispatch =====
    if getattr(mod.options, "export_tech_dispatch", False):
        periods, techs, disp = _dispatch_table(mod)
        export.write_table(
            mod,
            techs,
            output_file=os.path.join(summaries_dir, "dispatch_by_tech_periods.csv"),
            headings=("gentech",) + tuple(periods),
            values=lambda m, gt: (gt,)
            + tuple(disp[techs.index(gt)][j] for j in range(len(periods))),
        )

    # ===== CSV: transmission =====
    if getattr(mod.options, "export_transmission", False) and hasattr(
        mod, "TRANSMISSION_LINES"
    ):
        export.write_table(
            mod,
            mod.TRANSMISSION_LINES,
            output_file=os.path.join(summaries_dir, "tx_nameplate_by_path_periods.csv"),
            headings=("path",) + tuple(mod.PERIODS),
            values=lambda m, tx: (tx,)
            + tuple(m.TxCapacityNameplate[tx, p] for p in m.PERIODS),
        )
        if hasattr(mod, "BuildTx"):
            export.write_table(
                mod,
                mod.TRANSMISSION_LINES,
                output_file=os.path.join(
                    summaries_dir, "tx_builds_by_path_periods.csv"
                ),
                headings=("path",) + tuple(mod.PERIODS),
                values=lambda m, tx: (tx,)
                + tuple(
                    m.BuildTx[tx, p] if (tx, p) in m.BuildTx else 0.0 for p in m.PERIODS
                ),
            )

    # ===== PNG: capacity plot =====
    if getattr(mod.options, "plot_capacities", False):
        periods, techs, cap = _capacity_table(mod)
        _stacked_bar(
            periods,
            techs,
            cap,
            title="Installed capacity by technology and period",
            ylabel="MW",
            outfile=os.path.join(plots_dir, "capacity_by_tech_periods.png"),
        )

    # ===== PNG: dispatch plot =====
    if getattr(mod.options, "plot_tech_dispatch", False):
        periods, techs, disp = _dispatch_table(mod)
        _stacked_bar(
            periods,
            techs,
            disp,
            title="Total dispatched energy by technology and period",
            ylabel="MWh",
            outfile=os.path.join(plots_dir, "dispatch_by_tech_periods.png"),
        )
    # ===== Auto-run registered @graph functions =====
    if getattr(mod.options, "auto_graphs", False):
        rc = _run_graphs_cli(
            outdir=outdir,
            graphs=getattr(mod.options, "graphs", "all"),
            fmt=getattr(mod.options, "graph_format", "png"),
        )
        if rc != 0:
            # Don't crash the solve; just warn
            sys.stderr.write(
                "[graphs] Warning: auto-graphs runner returned non-zero exit code.\n"
            )
    _auto_run_registered_graphs(mod, outdir)
