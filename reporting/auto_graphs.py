# Auto-run all registered @graph plots after solve, saving PNGs into <outputs>/Graphs
# No flags required. If the graph tool is missing, this silently does nothing.

import os
import contextlib
import argparse


def _try_import_graph_api():
    try:
        from switch_model.tools.graph.main import Scenario
        from switch_model.tools.graph.cli import add_arguments, graph_scenarios_from_cli

        return Scenario, add_arguments, graph_scenarios_from_cli
    except Exception:
        return None, None, None


@contextlib.contextmanager
def _pushd(path):
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def post_solve(mod, outdir):
    # run all @graph functions after solve, saving into <outputs>/Graphs
    from switch_model.tools.graph.main import Scenario, graph_scenarios
    import os

    # 1) Make sure the module(s) that define @graph are imported so they register
    __import__("switch_model.generators.core.dispatch")
    # ^ add more modules here if you have other files with @graph functions:
    # __import__("your.package.with.graphs")

    # 2) Run the graph engine in-process
    os.makedirs(os.path.join(outdir, "Graphs"), exist_ok=True)
    scenarios = [Scenario(rel_path=".", name="")]
    # We call with module_names=None so it reads modules.txt, but the explicit import above
    # already registered your graphs even if modules.txt doesn't list them.
    with _pushd(outdir):
        graph_scenarios(
            scenarios=scenarios,
            graph_dir="Graphs",
            overwrite=True,
            module_names=None,  # let it use modules.txt; our explicit __import__ already registered your graphs
            skip_long=False,
        )
