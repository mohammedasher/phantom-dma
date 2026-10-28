"""Figs. 5-7: degradation modes for a reference and an alternative analysis.

Each of six panels shows one quantity for both analyses against throughput,
with a strip beneath giving the difference (alternative minus reference), in per cent of the
beginning-of-life quantity, at every RPT. Throughput: measured for the LG M50T
(Kirkaldy et al.), estimated for the P45B as cycles x twice the mean capacity.

Error bars on the traces: combined uncertainty from uncertainty.csv where it
exists (cell half-range and half-cell Monte Carlo in quadrature), otherwise the
cell half-range alone. Error bars on the strip: paired uncertainty of the difference (half-range across
cells of the within-cell difference and Monte Carlo SD of the within-draw
difference, in quadrature), since both analyses share cells, raw curves and draws.

Needs: the results folder; for Fig. 5 also the LG M50T summary files (paths.py).
Run:   python figures/fig_bias_modes.py 5     (or 6, or 7)
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from phantom_dma.data import m50t_throughput_kah
from phantom_dma.uncertainty import paired_difference

NAVY, POOL, SKY, GREY = "#003E74", "#0091D4", "#8CC4E8", "#5B6470"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.9, "axes.edgecolor": "#3B4450"})

MODES = [("SOH_pct", "SOH (%)"), ("LAM_PE_pct", "LAM$_{PE}$ (%)"), ("LLI_pct", "LLI (%)"),
         ("LAM_NE_pct", "LAM$_{NE}$ (%)"), ("LAM_Gr_pct", "LAM$^{*}_{Gr}$ (%)"), ("LAM_Si_pct", "LAM$^{*}_{Si}$ (%)")]
CONSTRAINED = {"PE": "LAM_PE_pct", "NE": "LAM_NE_pct", "LI": "LLI_pct", "LLI": "LLI_pct"}

FIGS = {
    5: dict(ref="M50T_discharge_IR-corrected", alt="M50T_discharge_uncorrected",
            labels=("IR-corrected", "no IR correction"), cell="M50T", rpts=(0, 2, 4, 6, 8, 10, 12, 14)),
    6: dict(ref="P45B_discharge_2.5-4.2", alt="P45B_discharge_3.0-4.2",
            labels=("full window  2.5–4.2 V", "window cut to 3.0 V"), cell="P45B", rpts=None),
    7: dict(ref="P45B_discharge_2.94-4.2", alt="P45B_charge_2.94-4.2",
            labels=("discharge branch", "charge branch"), cell="P45B", rpts=None),
}


P45B_CAPACITY_RUN = "P45B_discharge_2.5-4.2"   # one capacity source for every P45B figure


def throughput(results, cell_type):
    """Mean cumulative throughput (kAh) of the two cells at each RPT."""
    if cell_type == "M50T":
        return pd.concat([m50t_throughput_kah("D"), m50t_throughput_kah("E")], axis=1).mean(axis=1)
    # P45B: estimated as the cycle count times twice the mean measured capacity
    # at the interval endpoints (no throughput log is available for these cells).
    summary = pd.read_csv(os.path.join(results, P45B_CAPACITY_RUN, "summary.csv"))
    out = []
    for cell, g in summary.groupby("Cell"):
        g = g.sort_values("RPT")
        q, n = g.Q_exp_Ah.values, np.diff(g.RPT.values)
        out.append(pd.Series(np.r_[0, np.cumsum(n * (q[:-1] + q[1:]))] / 1000, index=g.RPT.values))
    return pd.concat(out, axis=1).mean(axis=1)


def load(results, run, rpts):
    s = pd.read_csv(os.path.join(results, run, "summary.csv"))
    if rpts is not None:
        s = s[s.RPT.isin(rpts)]
    g = s.groupby("RPT")
    mean = g.mean(numeric_only=True)
    half = (g.max(numeric_only=True) - g.min(numeric_only=True)) / 2
    unc = half.copy()
    u_path = os.path.join(results, run, "uncertainty.csv")
    if os.path.exists(u_path):
        u = pd.read_csv(u_path).set_index("RPT")
        for key, _ in MODES:
            if key + "_combined" in u.columns:
                mean[key] = u[key + "_mean"].reindex(mean.index)
                unc[key] = u[key + "_combined"].reindex(mean.index)
    held = {k: set() for k in CONSTRAINED.values()}
    for _, row in s.dropna(subset=["constraint_active"]).iterrows():
        for tag in str(row.constraint_active).replace(",", " ").split():
            if tag in CONSTRAINED:
                held[CONSTRAINED[tag]].add(row.RPT)
    return s, mean, unc, held


def paired(results, ref, alt, key):
    """Paired uncertainty of the difference (cells and Monte Carlo draws), per RPT."""
    return paired_difference(results, ref, alt, key)["combined"]


def make(n, results, out):
    cfg = FIGS[n]
    sa, A, ua, ha = load(results, cfg["ref"], cfg["rpts"])
    sb, B, ub, hb = load(results, cfg["alt"], cfg["rpts"])
    rpts = A.index.values
    x = throughput(results, cfg["cell"]).reindex(rpts).values
    span = float(x[-1] - x[0])
    bw = 0.065 * span

    fig = plt.figure(figsize=(7.4, 5.0), dpi=300)
    outer = GridSpec(2, 3, figure=fig, left=0.085, right=0.988, top=0.975, bottom=0.085, wspace=0.34, hspace=0.34)
    for k, (key, ylab) in enumerate(MODES):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // 3, k % 3], height_ratios=[3.0, 1.15], hspace=0.10)
        ax, rs = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        a, b = A[key].values, B[key].values
        ea, eb = ua[key].values, ub[key].values

        ax.fill_between(x, a, b, color=SKY, alpha=0.40, lw=0, zorder=2)
        for vals, err, held, colour, ls, mk in ((a, ea, ha, NAVY, "-", "o"), (b, eb, hb, POOL, "--", "s")):
            ax.errorbar(x, vals, yerr=err, color=colour, lw=1.6, ls=ls, marker="none", elinewidth=1.0, capsize=2.0, zorder=5)
            ax.plot(x, vals, mk, color=colour, ms=3.0, zorder=6)
        ax.set_ylabel(ylab, fontsize=8.4, labelpad=2)
        ax.tick_params(labelsize=7.0, length=2.2, labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#EEF1F4", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.055 * span, x[-1] + 0.06 * span)

        d = b - a
        du = paired(results, cfg["ref"], cfg["alt"], key).reindex(rpts).fillna(0).values
        rs.axhline(0, color="#3B4450", lw=0.9, zorder=4)
        rs.bar(x, d, bw, color=POOL, alpha=0.85, zorder=3, yerr=du,
               error_kw=dict(ecolor="#25516F", elinewidth=0.9, capsize=1.8))
        lim = max(np.nanmax(np.abs(d) + du), 0.5)
        pad = 0.07 * lim
        last_x, stagger = -np.inf, 0.0
        for xi, di, ei in zip(x, d, du):
            if abs(di) < 0.05:
                continue
            # lift the label of a point that sits close to the previous labelled one
            stagger = 0.45 * lim if (xi - last_x) < 0.13 * span and stagger == 0.0 else 0.0
            last_x = xi
            yt = di + (ei + pad + stagger if di > 0 else -(ei + pad + stagger))
            rs.text(xi, yt, f"{di:+.1f}", fontsize=5.6, color=POOL, ha="center",
                    va="bottom" if di > 0 else "top", zorder=5)
        rs.set_ylim(-1.75 * lim, 1.75 * lim)
        rs.set_xlim(-0.055 * span, x[-1] + 0.06 * span)
        rs.set_ylabel("Δ (%)", fontsize=6.4, labelpad=2)
        rs.tick_params(labelsize=6.4, length=2.0)
        rs.set_yticks([])
        rs.spines[["top", "right", "left"]].set_visible(False)
        if k // 3 == 1:
            rs.set_xlabel("throughput (kAh)", fontsize=8.0, labelpad=2)
        ax.text(0.015, 1.015, f"({'abcdef'[k]})", transform=ax.transAxes, fontsize=7.8, color="#3B4450",
                fontweight="bold", va="bottom")
        if k == 0:
            h = [plt.Line2D([], [], color=NAVY, lw=1.6, marker="o", ms=3.2, label=cfg["labels"][0]),
                 plt.Line2D([], [], color=POOL, lw=1.6, ls="--", marker="s", ms=3.2, label=cfg["labels"][1])]
            ax.legend(handles=h, fontsize=7.0, frameon=False, loc="lower left", handlelength=1.8,
                      labelspacing=0.22, borderpad=0.1)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out, "| last-RPT differences (%):",
          {key: round(float(B[key].values[-1] - A[key].values[-1]), 2) for key, _ in MODES})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fig", type=int, choices=sorted(FIGS))
    p.add_argument("--results", default=paths.RESULTS)
    p.add_argument("--out")
    a = p.parse_args()
    names = {5: "Fig5_IR_correction_bias.png", 6: "Fig6_window_bias.png", 7: "Fig7_branch_bias.png"}
    out = a.out or os.path.join(paths.OUTPUT, "figures", names[a.fig])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make(a.fig, a.results, out)
