"""Step 1: fit the degradation modes of every RPT for one or more analyses.

Examples
    python step1_fit.py P45B_discharge_2.5-4.2        # one analysis, both cells
    python step1_fit.py --all                         # every analysis in the paper
    python step1_fit.py --all --jobs 4                # four cell fits at a time

Each analysis is fitted for both cells. One cell takes 20 to 50 minutes on one
processor core. Results go to results/<analysis>/: one summary table per cell
(cell1_summary.csv, cell2_summary.csv), the two combined (summary.csv), and the
measured and fitted curves of every RPT (fits/<cell>_RPT_<nnn>.npz).
The repository already contains these results, so this step is optional.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import paths
from phantom_dma import fit
from phantom_dma.analyses import ANALYSES, CELLS, HALFCELLS
from phantom_dma.data import load_curves
from phantom_dma.model import CellModel, losses


def fit_one_cell(task):
    """Fit every RPT of one cell for one analysis and save the results."""
    name, cell = task
    settings = ANALYSES[name]
    start = time.time()
    fit.set_bounds(settings.get("bounds"))
    model = CellModel(*HALFCELLS[(settings["cell"], settings["branch"])])
    curves = load_curves(settings, cell)
    splits = None
    if settings.get("split_from"):
        # transition point taken, RPT by RPT, from the nominal fits of another analysis of the same cell
        src = os.path.join(paths.RESULTS, settings["split_from"], "fits")
        splits = {r: float(np.load(os.path.join(src, f"{cell}_RPT_{r:03d}.npz"))["delta"]) for r in curves}
    table, arrays = fit.fit_sequence(model, curves, settings["branch"], settings["window"],
                                     splits=splits,
                                     log=lambda s: print(f"[{name} {cell}]{s}", flush=True))

    # Degradation modes relative to the first RPT (beginning of life).
    first = table.iloc[0]
    capacities0 = {k: first[k + "_Ah"] for k in ("Q_NE", "Q_PE", "Q_LI", "Q_Si", "Q_Gr")}
    for i, row in table.iterrows():
        capacities = {k: row[k + "_Ah"] for k in capacities0}
        for mode, value in losses(capacities, capacities0).items():
            table.loc[i, mode] = value
    table.insert(0, "Cell", cell)
    table["SOH_pct"] = 100 * table["Q_exp_Ah"] / first["Q_exp_Ah"]

    folder = os.path.join(paths.RESULTS, name)
    os.makedirs(os.path.join(folder, "fits"), exist_ok=True)
    table.to_csv(os.path.join(folder, f"{cell}_summary.csv"), index=False)
    for rpt, a in arrays.items():
        np.savez(os.path.join(folder, "fits", f"{cell}_RPT_{rpt:03d}.npz"), **a)
    print(f"[{name} {cell}] done in {(time.time() - start) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analyses", nargs="*", help="analysis names (see phantom_dma/analyses.py)")
    parser.add_argument("--all", action="store_true", help="fit every analysis")
    parser.add_argument("--jobs", type=int, default=1, help="number of cell fits to run at once")
    args = parser.parse_args()
    names = list(ANALYSES) if args.all else args.analyses
    if not names:
        parser.error("give one or more analysis names, or --all")

    tasks = [(name, cell) for name in names for cell in CELLS[ANALYSES[name]["cell"]]]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(fit_one_cell, tasks)
    else:
        for task in tasks:
            fit_one_cell(task)

    for name in names:
        folder = os.path.join(paths.RESULTS, name)
        cells = CELLS[ANALYSES[name]["cell"]]
        pd.concat([pd.read_csv(os.path.join(folder, f"{c}_summary.csv")) for c in cells]).to_csv(
            os.path.join(folder, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
