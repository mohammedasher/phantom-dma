"""Step 2: half-cell offset Monte Carlo, which gives the uncertainty of each mode.

Run step 1 first for the same analyses (or use the results already in the repository).

Examples
    python step2_monte_carlo.py P45B_discharge_2.5-4.2
    python step2_monte_carlo.py --all --jobs 4

Each draw shifts the three half-cell curves by random offsets (standard deviation
2 mV) and refits every RPT. 50 draws per cell take several hours per analysis.
Results go to results/<analysis>/: cell1_mc.csv and cell2_mc.csv (every draw) and
uncertainty.csv (mean of the two cells, half their range, Monte Carlo standard
deviation and the two combined).
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import paths
from phantom_dma import fit
from phantom_dma.analyses import ANALYSES, CELLS, HALFCELLS, WITH_MONTE_CARLO
from phantom_dma.data import load_curves
from phantom_dma.uncertainty import combine, run_mc


def monte_carlo_one_cell(task):
    name, cell, index = task
    settings = ANALYSES[name]
    fit.set_bounds(settings.get("bounds"))   # the search limits of the nominal fit
    folder = os.path.join(paths.RESULTS, name)
    nominal = pd.read_csv(os.path.join(folder, f"{cell}_summary.csv"))
    curves = load_curves(settings, cell)
    if settings.get("mc_rpts"):
        # Only the RPTs that are compared with another analysis.
        curves = {r: c for r, c in curves.items() if r in settings["mc_rpts"]}
    # the same split as the nominal fit of each RPT
    splits = {r: float(np.load(os.path.join(folder, "fits", f"{cell}_RPT_{r:03d}.npz"))["delta"]) for r in curves}
    draws = run_mc(HALFCELLS[(settings["cell"], settings["branch"])], curves, settings["branch"],
                   settings["window"], nominal, cell_index=index,
                   log=lambda s: print(f"[{name} {cell}]{s}", flush=True), splits=splits)
    draws.insert(0, "Cell", cell)
    draws.to_csv(os.path.join(folder, f"{cell}_mc.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analyses", nargs="*")
    parser.add_argument("--all", action="store_true", help="every analysis that has a Monte Carlo in the paper")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    names = WITH_MONTE_CARLO if args.all else args.analyses
    if not names:
        parser.error("give one or more analysis names, or --all")

    # index 0 or 1 gives each cell its own random-number seed
    tasks = [(n, c, i) for n in names for i, c in enumerate(CELLS[ANALYSES[n]["cell"]])]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(monte_carlo_one_cell, tasks)
    else:
        for task in tasks:
            monte_carlo_one_cell(task)

    for name in names:
        folder = os.path.join(paths.RESULTS, name)
        cells = CELLS[ANALYSES[name]["cell"]]
        summary = pd.read_csv(os.path.join(folder, "summary.csv"))
        draws = pd.concat([pd.read_csv(os.path.join(folder, f"{c}_mc.csv")) for c in cells])
        combine(summary, draws).to_csv(os.path.join(folder, "uncertainty.csv"), index=False)


if __name__ == "__main__":
    main()
