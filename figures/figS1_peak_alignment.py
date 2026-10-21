"""SI figure: one representative fit on each branch, same cell, RPT and voltage range.

Shows that the cell-voltage RMSE does not rank fits by peak alignment: the
charge fit has the lower RMSE but displaces a peak, while the discharge fit
places every peak to within a few mV.

Needs: results/P45B_discharge_2.94-4.2 and results/P45B_charge_2.94-4.2.
Run:   python figures/figS1_peak_alignment.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phantom_dma.peaks import find_ic_peaks, ic_curve

NAVY, GREY = "#003E74", "#6B7480"
RUNS = [("P45B_discharge_2.94-4.2", "discharge"), ("P45B_charge_2.94-4.2", "charge")]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.9, "axes.edgecolor": "#3B4450"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=paths.RESULTS)
    p.add_argument("--cell", default="cell1")
    p.add_argument("--rpt", type=int, default=500)
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "FigS1_peak_alignment.png"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300, sharey=True)
    for k, (run, label) in enumerate(RUNS):
        f = np.load(os.path.join(a.results, run, "fits", f"{a.cell}_RPT_{a.rpt:03d}.npz"))
        rmse = pd.read_csv(os.path.join(a.results, run, f"{a.cell}_summary.csv")).set_index("RPT").loc[a.rpt, "RMSE_mV"]
        z, u, vm = f["z"], f["U_exp"], f["V_model"]
        vu, icu = ic_curve(z, u)
        vv, icm = ic_curve(z, vm)
        prom = 0.15 * np.nanmax(np.abs(icu[np.isfinite(icu)]))
        pu, pm = find_ic_peaks(icu, prom), find_ic_peaks(icm, 0.5 * prom)
        ax = axs[k]
        ax.plot(vu, np.abs(icu), color=GREY, lw=1.2, label="measured")
        ax.plot(vv, np.abs(icm), color=NAVY, lw=1.0, ls="--", label="model")
        top = np.nanpercentile(np.abs(icu[np.isfinite(icu)]), 99.5)
        for i in pu:
            j = pm[np.argmin(np.abs(vv[pm] - vu[i]))]
            dv = 1000 * (vv[j] - vu[i])
            ax.annotate(f"{dv:+.0f} mV", (vu[i], min(abs(icu[i]), 1.1 * top)), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5, color=NAVY)
        ax.set_ylim(0, 1.3 * top)
        ax.set_title(f"({'ab'[k]}) {label}: RMSE {rmse:.1f} mV", fontsize=8, loc="left")
        ax.set_xlabel("cell voltage (V)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#EEF1F4", lw=0.7); ax.set_axisbelow(True)
    axs[0].set_ylabel(r"$|\mathrm{d}z_{\mathrm{FC}}/\mathrm{d}V|$ (V$^{-1}$)", fontsize=8)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
