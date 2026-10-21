"""Fig. 2: pulse resistance R_0.1s against SOC and ageing, for both cell types.

(a) LG M50T. Each column is one RPT; each tile is one GITT pulse and spans the
    SOC interval that pulse covered. The colour gives R_0.1s relative to the
    mean at 30-70 % SOC at the first RPT. The dashed line marks where R_0.1s
    starts to rise at low SOC (R above the mid-SOC mean + 3 standard
    deviations); its error bars span both cells and thresholds of 2 to 4
    standard deviations. Hatched: pulses cut short by the 2.5 V limit (not shown).
(b) Molicel P45B. R_0.1s of the 1C pulses at 70, 50 and 30 % SOC, same colours,
    at all seven RPTs (0 to 500 cycles).

Every tile is the mean of the two cells. The first row above each panel gives the
state of health: the low-rate pOCV discharge capacity (C/10 for the LG M50T, C/20
for the P45B) relative to the first RPT, mean of the two cells. This is the
capacity the degradation modes are fitted to (Figs. 5 and 6). The second row
gives half the range of the two cells (the largest value in that column).

Needs: the LG M50T GITT files and the P45B data (paths.py).
Run:   python figures/fig2_resistance.py
Writes output/figures/Fig2_resistance.png, and the plotted numbers in
       Fig2_values_M50T.csv and Fig2_values_P45B.csv next to it.
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from phantom_dma.data import M50T_PULSE_RPTS, m50t_curve, p45b_curve
from phantom_dma.resistance import m50t_pulses, p45b_pulse_resistance

# LG M50T: cell2 ("E" in Kirkaldy et al.) is listed first; it is the cell used in Fig. 3.
M50T_CELLS = {"cell2": "E", "cell1": "D"}
P45B_CELLS = ["cell1", "cell2"]
P45B_RPTS = [0, 50, 100, 200, 300, 400, 500]   # RPTs shown in panel (b)
P45B_PULSE_SOC = [70, 50, 30]             # SOC (%) of the three pulses, in test order
CYCLES_PER_M50T_RPT = 78                  # ageing cycles between two M50T tests
ONSET_SIGMA = 3.0                         # threshold for the low-SOC onset (line)
ONSET_SIGMA_RANGE = [2.0, 2.5, 3.0, 3.5, 4.0]   # thresholds spanned by the error bars

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
                     "text.color": "k", "axes.labelcolor": "k"})

# Colour scale: R_0.1s relative to the cell's own first-RPT mid-SOC mean.
BINS = np.array([0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
CMAP = ListedColormap(["#EAF7FC", "#B7E3F4", "#6CC4EA", "#1E9BD7", "#1868B7", "#1C3F8F", "#141F5C"])
CMAP.set_under("#FFFFFF")
CMAP.set_over("#0A0F33")
NORM = BoundaryNorm(BINS, CMAP.N)
TILE_WIDTH = 0.84


def pocv_capacity(cell_type, cell, rpt):
    """Capacity (Ah) of the low-rate pOCV discharge: C/10 for the LG M50T, C/20 for the P45B."""
    if cell_type == "M50T":
        return m50t_curve(cell, rpt)["Q"].max()
    return p45b_curve(cell, rpt, "discharge")["Q"].max()


def state_of_health(cell_type, cells, rpts):
    """pOCV capacity relative to the first RPT, mean of the cells, per RPT."""
    q = {r: np.mean([pocv_capacity(cell_type, c, r) for c in cells]) for r in rpts}
    return {r: q[r] / q[rpts[0]] for r in rpts}


def low_soc_onset(pulses, n, n_sigma):
    """Highest pulse SOC below 50 % at which R exceeds the 30-70 % SOC mean by n_sigma
    standard deviations. Uses the first n pulses, full pulses only."""
    full = pulses["full"][:n]
    soc, r = pulses["s0"][:n][full], pulses["R"][:n][full]
    mid = r[(soc >= 30) & (soc <= 70)]
    high = soc[(r > mid.mean() + n_sigma * mid.std(ddof=1)) & (soc < 50)]
    return high.max() if len(high) else np.nan


def m50t_two_cells():
    """Per RPT: the mean of the two M50T cells, pulse by pulse, and the onset of the
    low-SOC rise. Pulses are matched by their index (both cells ran the same
    protocol); only pulses that are full in both cells are kept.

    The onset is found on each cell's own R(SOC) and then averaged: averaging the
    two curves first would smooth the rise and move the crossing.
    """
    (name_a, cell_a), (name_b, cell_b) = M50T_CELLS.items()
    out = {}
    for rpt in M50T_PULSE_RPTS:
        a, b = m50t_pulses(cell_a, rpt), m50t_pulses(cell_b, rpt)
        n = min(len(a["R"]), len(b["R"]))
        keep = a["full"][:n] & b["full"][:n]
        s0 = ((a["s0"][:n] + b["s0"][:n]) / 2)[keep]
        s1 = ((a["s1"][:n] + b["s1"][:n]) / 2)[keep]
        r = ((a["R"][:n] + b["R"][:n]) / 2)[keep]
        half_range = (np.abs(a["R"][:n] - b["R"][:n]) / 2)[keep]

        onset_3 = [low_soc_onset(p, n, ONSET_SIGMA) for p in (a, b)]
        onset_all = [low_soc_onset(p, n, k) for p in (a, b) for k in ONSET_SIGMA_RANGE]
        onset_all = [v for v in onset_all if np.isfinite(v)]

        out[rpt] = dict(s0=s0, s1=s1, R=r, half=half_range,
                        onset=float(np.nanmean(onset_3)) if np.any(np.isfinite(onset_3)) else 0,
                        onset_lo=min(onset_all) if onset_all else 0,
                        onset_hi=max(onset_all) if onset_all else 0,
                        mid_mean=r[(s0 >= 30) & (s0 <= 70)].mean(),
                        R_cells={name_a: a["R"][:n][keep], name_b: b["R"][:n][keep]},
                        onset_cells={name_a: onset_3[0], name_b: onset_3[1]})
    return out


def draw_m50t(ax, m50t, r_ref):
    """Panel (a): one column of tiles per RPT, the onset line and its error bars."""
    for k, rpt in enumerate(M50T_PULSE_RPTS):
        d = m50t[rpt]
        lowest = d["s1"].min()
        ax.add_patch(Rectangle((k - TILE_WIDTH / 2, 0), TILE_WIDTH, lowest, fc="none", ec="0.6",
                               hatch="////", lw=0, zorder=1))
        for top, bottom, r in zip(d["s0"], d["s1"], d["R"]):
            ax.add_patch(Rectangle((k - TILE_WIDTH / 2, bottom), TILE_WIDTH, top - bottom,
                                   fc=CMAP(NORM(r / r_ref)), ec="white", lw=0.4, zorder=2))
    # Onset as a step line, one level per column.
    xs = np.array([[k - TILE_WIDTH / 2, k + TILE_WIDTH / 2] for k in range(len(M50T_PULSE_RPTS))]).ravel()
    ys = np.repeat([m50t[r]["onset"] for r in M50T_PULSE_RPTS], 2)
    ax.plot(xs, ys, c="k", lw=1.4, ls="--", zorder=5)
    # One error bar per RPT, spanning both cells and 2-4 sigma.
    xc = np.arange(len(M50T_PULSE_RPTS))
    yc = np.array([m50t[r]["onset"] for r in M50T_PULSE_RPTS])
    lo = np.array([m50t[r]["onset_lo"] for r in M50T_PULSE_RPTS])
    hi = np.array([m50t[r]["onset_hi"] for r in M50T_PULSE_RPTS])
    ax.errorbar(xc, yc, yerr=[np.clip(yc - lo, 0, None), np.clip(hi - yc, 0, None)], fmt="none",
                ecolor="k", elinewidth=1.1, capsize=3.0, capthick=1.1, zorder=6)


def draw_p45b(ax, p45b_mean, r_ref):
    """Panel (b): three tiles per RPT, labelled with R_0.1s in mOhm."""
    for k, rpt in enumerate(P45B_RPTS):
        for soc, r in zip(P45B_PULSE_SOC, p45b_mean[rpt]):
            ax.add_patch(Rectangle((k - TILE_WIDTH / 2, soc - 7), TILE_WIDTH, 14,
                                   fc=CMAP(NORM(r / r_ref)), ec="white", lw=0.4, zorder=2))
            ax.text(k, soc, f"{r:.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if r / r_ref >= 1.05 else "black", zorder=3)


def label_axes(ax, x_labels, soh, spread, title):
    """Bottom axis (cycles), the two rows above the panel (SOH and cell-to-cell range) and the title."""
    n = len(x_labels)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="x", pad=3, length=0)
    ax.set_xlabel("RPT after ageing cycles", fontsize=8.5)

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(range(n))
    top.set_xticklabels([f"{v:.2f}" for v in soh], fontsize=7.3)
    top.tick_params(length=0)
    top.set_xlabel("State of health", fontsize=8)

    top2 = ax.twiny()
    top2.set_xlim(ax.get_xlim())
    top2.spines["top"].set_position(("outward", 27))
    for sp in top2.spines.values():
        sp.set_visible(False)
    top2.set_xticks(range(n))
    top2.set_xticklabels([f"±{v:.2f}" for v in spread], fontsize=7.0, color="0.3")
    top2.tick_params(length=0)
    top2.set_xlabel("Cell-to-cell range (m$\\Omega$)", fontsize=8, color="0.3")

    ax.text(0, 1.33, title, transform=ax.transAxes, fontweight="bold", fontsize=9.5, va="bottom")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def r_span_label(lo, hi, fmt):
    return f"$\\mathbf{{R_{{0.1\\,s}}}}$ ≈ {lo:{fmt}}–{hi:{fmt}} m$\\mathbf{{\\Omega}}$"


def make_figure(m50t, m50t_soh, p45b_mean, p45b_half, p45b_soh, out):
    m50t_ref = m50t[0]["mid_mean"]
    p45b_ref = p45b_mean[0].mean()

    fig = plt.figure(figsize=(9.6, 4.6))
    ax_a = fig.add_axes([0.07, 0.22, 0.50, 0.64])
    ax_b = fig.add_axes([0.61, 0.22, 0.26, 0.64], sharey=ax_a)
    cax = fig.add_axes([0.925, 0.22, 0.015, 0.64])

    draw_m50t(ax_a, m50t, m50t_ref)
    draw_p45b(ax_b, p45b_mean, p45b_ref)

    r_max_m50t = max(m50t[r]["R"].max() for r in M50T_PULSE_RPTS)
    label_axes(ax_a, [str(CYCLES_PER_M50T_RPT * r) for r in M50T_PULSE_RPTS],
               [m50t_soh[r] for r in M50T_PULSE_RPTS],
               [max(m50t[r]["half"]) for r in M50T_PULSE_RPTS],
               "(a) LG M50T   " + r_span_label(m50t_ref, r_max_m50t, ".0f"))
    label_axes(ax_b, [str(n) for n in P45B_RPTS],
               [p45b_soh[n] for n in P45B_RPTS],
               [max(p45b_half[n]) for n in P45B_RPTS],
               "(b) Molicel P45B   " + r_span_label(min(p45b_mean[n].min() for n in P45B_RPTS),
                                                    max(p45b_mean[n].max() for n in P45B_RPTS), ".1f"))

    ax_a.set_ylim(0, 100)
    ax_a.set_yticks([0, 20, 40, 60, 80, 100])
    ax_a.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    ax_a.set_ylabel("State of charge during pulse", fontsize=8.5)
    plt.setp(ax_b.get_yticklabels(), visible=False)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=BINS,
                      extend="both", extendfrac=0.04)
    cb.ax.set_yticklabels([f"{100 * (b - 1):+.0f}%".replace("+0%", "0%") for b in BINS], fontsize=7.3)
    cax.set_title("$R_{0.1\\,\\mathrm{s}}$ change vs\nown BoL mid-SOC", fontsize=7.5, pad=8)

    onset_handle = (Line2D([], [], c="k", ls="--", lw=1.4), Line2D([], [], c="k", lw=1.1, marker="_", ms=5, ls="-"))
    ax_a.legend(handles=[onset_handle, Patch(fc="none", ec="0.6", hatch="////")],
                labels=[r"onset of low-SOC rise ($R$ > mid-SOC mean + 3$\sigma$); bars span both cells and 2–4$\sigma$",
                        "pulses truncated by the 2.5 V cut-off (omitted)"],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                loc="upper left", bbox_to_anchor=(-0.01, -0.1), frameon=False, fontsize=7.2,
                handlelength=2.4, labelspacing=0.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)


def m50t_table(m50t, m50t_soh):
    """The numbers behind panel (a), one row per RPT."""
    rows = []
    for rpt in M50T_PULSE_RPTS:
        d = m50t[rpt]
        mid = (d["s0"] >= 30) & (d["s0"] <= 70)
        r1, r2 = d["R_cells"]["cell1"][mid].mean(), d["R_cells"]["cell2"][mid].mean()
        rows.append(dict(RPT=rpt, cycles=CYCLES_PER_M50T_RPT * rpt, SOH=m50t_soh[rpt],
                         R_midSOC_cell1_mOhm=r1, R_midSOC_cell2_mOhm=r2,
                         R_midSOC_mean_mOhm=d["R"][mid].mean(),
                         R_midSOC_half_range_mOhm=abs(r1 - r2) / 2,
                         R_min_over_SOC_mOhm=d["R"].min(), R_max_over_SOC_mOhm=d["R"].max(),
                         cell_to_cell_max_half_range_mOhm=max(d["half"]),
                         onset_cell1_pctSOC=d["onset_cells"]["cell1"],
                         onset_cell2_pctSOC=d["onset_cells"]["cell2"],
                         onset_plotted_pctSOC=d["onset"],
                         onset_bar_low_pctSOC=d["onset_lo"], onset_bar_high_pctSOC=d["onset_hi"]))
    return pd.DataFrame(rows).round(3)


def p45b_table(p45b, p45b_mean, p45b_half, p45b_soh):
    """The numbers behind panel (b), one row per tile."""
    rows = []
    for rpt in P45B_RPTS:
        for j, soc in enumerate(P45B_PULSE_SOC):
            rows.append(dict(RPT=rpt, SOC_pct=soc,
                             R_cell1_mOhm=p45b["cell1"][rpt][j], R_cell2_mOhm=p45b["cell2"][rpt][j],
                             R_mean_mOhm=p45b_mean[rpt][j], R_half_range_mOhm=p45b_half[rpt][j],
                             SOH=p45b_soh[rpt]))
    return pd.DataFrame(rows).round(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "Fig2_resistance.png"))
    a = p.parse_args()
    folder = os.path.dirname(a.out)
    os.makedirs(folder, exist_ok=True)

    m50t = m50t_two_cells()
    m50t_soh = state_of_health("M50T", M50T_CELLS.values(), M50T_PULSE_RPTS)

    # P45B: tiles are the two-cell mean.
    p45b_soh = state_of_health("P45B", P45B_CELLS, P45B_RPTS)
    p45b = {c: {n: p45b_pulse_resistance(c, n) for n in P45B_RPTS} for c in P45B_CELLS}
    p45b_mean = {n: (p45b["cell1"][n] + p45b["cell2"][n]) / 2 for n in P45B_RPTS}
    p45b_half = {n: np.abs(p45b["cell1"][n] - p45b["cell2"][n]) / 2 for n in P45B_RPTS}

    make_figure(m50t, m50t_soh, p45b_mean, p45b_half, p45b_soh, a.out)
    m50t_table(m50t, m50t_soh).to_csv(os.path.join(folder, "Fig2_values_M50T.csv"), index=False)
    p45b_table(p45b, p45b_mean, p45b_half, p45b_soh).to_csv(os.path.join(folder, "Fig2_values_P45B.csv"), index=False)

    print(f"wrote {a.out}")
    print("State of health (pOCV capacity, two-cell mean):")
    print("  M50T:", {r: round(float(m50t_soh[r]), 3) for r in M50T_PULSE_RPTS})
    print("  P45B:", {n: round(float(p45b_soh[n]), 3) for n in P45B_RPTS})
    print("M50T onset of the low-SOC rise (% SOC), mean of both cells at 3 sigma:",
          {r: round(m50t[r]["onset"]) for r in M50T_PULSE_RPTS})
    print("  error bar (both cells, 2-4 sigma):",
          {r: (round(m50t[r]["onset_lo"]), round(m50t[r]["onset_hi"])) for r in M50T_PULSE_RPTS})
    print("M50T cell-to-cell half-range (mOhm):", {r: round(float(max(m50t[r]["half"])), 3) for r in M50T_PULSE_RPTS})
    print("P45B two-cell mean at 70/50/30 % SOC (mOhm):", {n: [round(float(v), 2) for v in p45b_mean[n]] for n in P45B_RPTS})


if __name__ == "__main__":
    main()
