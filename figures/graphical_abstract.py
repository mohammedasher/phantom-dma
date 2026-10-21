"""Graphical abstract: three analysis choices change the fitted degradation of the same cells.

Left: C/20 charge and discharge of one P45B cell at RPT 0, marking the three choices
(1 resistance, 2 branch, 3 lower cut-off). Right: the change each choice makes to one
fitted loss at the last RPT, as a percentage of the recommended analysis (Table 2):
1 positive-electrode loss, LG M50T; 2 silicon loss, P45B; 3 negative-electrode loss, P45B.

Elsevier format: 1500 x 600 px at 300 dpi (5 x 2 in).
Needs: the results folder and the P45B data (paths.py).
Run:   python figures/graphical_abstract.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from phantom_dma.data import p45b_file

NAVY, GREY = "#003E74", "#5B6470"
IR_C, BR_C, WN_C = "#C2570A", "#0091D4", "#0E7C6B"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1, "axes.edgecolor": "#3B4450"})


def relative_change(ref, alt, key, rpt):
    """Change (%) of one mode, alternative against reference, as in Table 2."""
    a = pd.read_csv(os.path.join(paths.RESULTS, ref, "summary.csv"))
    b = pd.read_csv(os.path.join(paths.RESULTS, alt, "summary.csv"))
    if ref.startswith("M50T"):   # LG M50T: mean of the relative change over the ageing RPTs
        return np.mean([100 * (b[b.RPT == k][key].mean() / a[a.RPT == k][key].mean() - 1)
                        for k in sorted(set(a.RPT) & set(b.RPT)) if 0 < k <= rpt])
    return 100 * (b[b.RPT == rpt][key].mean() / a[a.RPT == rpt][key].mean() - 1)


def main():
    rows = [("1  no IR correction", "without vs with IR correction", r"phantom LAM$_\mathrm{PE}$", IR_C,
             relative_change("M50T_discharge_IR-corrected", "M50T_discharge_uncorrected", "LAM_PE_pct", 14)),
            ("2  charge branch", "charge vs discharge branch", "underestimated Si loss", BR_C,
             relative_change("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2", "LAM_Si_pct", 500)),
            ("3  reduced voltage window", "reduced vs full voltage window", r"phantom LAM$_\mathrm{NE}$", WN_C,
             relative_change("P45B_discharge_2.5-4.2", "P45B_discharge_3.0-4.2", "LAM_NE_pct", 500))]

    # C/20 charge and discharge of one P45B cell at RPT 0, on a common capacity axis
    d = pd.read_csv(p45b_file("cell2", 0))
    ch = d[(d["Step Index"] == 6) & (d["Current / A"] > 0.05)]
    dc = d[(d["Step Index"] == 8) & (d["Current / A"] < -0.05)]
    qd = dc["Discharge Capacity / Ah"].values
    qd = qd.max() - qd
    vd = dc["Voltage / V"].values
    o = np.argsort(qd)
    qd, vd = qd[o][::max(1, len(qd) // 500)], vd[o][::max(1, len(qd) // 500)]
    qc = ch["Charge Capacity / Ah"].values
    qc = qc + (qd.max() - qc.max())
    vc = ch["Voltage / V"].values
    qc, vc = qc[::max(1, len(qc) // 500)], vc[::max(1, len(qc) // 500)]

    fig = plt.figure(figsize=(5.0, 2.0), dpi=300)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.072, 0.195, 0.285, 0.640])
    # 3: voltage window, both ends (shaded bands removed by a narrower window)
    for lo, hi in ((2.45, 3.0), (4.0, 4.32)):
        ax.add_patch(Rectangle((-0.1, lo), 4.7, hi - lo, facecolor=WN_C, alpha=0.22, edgecolor="none", zorder=0))
    for v in (3.0, 4.0):
        ax.axhline(v, color=WN_C, lw=1.0, ls=":", zorder=1)
    # 2: hysteresis, charge above discharge at the same capacity
    ax.fill_between(qd, vd, np.interp(qd, qc, vc), color=BR_C, alpha=0.30, lw=0, zorder=2)
    ax.plot(qc, vc, color=BR_C, lw=2.0, ls="--", zorder=4)
    ax.plot(qd, vd, color=NAVY, lw=2.2, zorder=5)
    q2 = 0.30
    v_d, v_c = np.interp(q2, qd, vd), np.interp(q2, qc, vc)
    ax.annotate("", xy=(q2, v_c), xytext=(q2, v_d), zorder=7,
                arrowprops=dict(arrowstyle="<|-|>", color=BR_C, lw=1.5, mutation_scale=7, shrinkA=0, shrinkB=0))
    # 1: resistance, the same discharge curve shifted down under load (drawn exaggerated)
    shift = 0.24
    ax.plot(qd, vd - shift, color=IR_C, lw=1.6, zorder=3)
    q1 = 2.3
    v1 = np.interp(q1, qd, vd)
    ax.annotate("", xy=(q1, v1 - shift), xytext=(q1, v1), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=IR_C, lw=2.2, mutation_scale=11, shrinkA=0, shrinkB=0))
    ax.set_xlim(-0.1, 4.6)
    ax.set_ylim(2.45, 4.32)
    ax.set_xlabel("capacity (Ah)", fontsize=8.0, labelpad=1.5)
    ax.set_ylabel("voltage (V)", fontsize=8.0, labelpad=1.5)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([2.5, 3.0, 3.5, 4.0])
    ax.tick_params(labelsize=7.2, length=2.2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    def badge(x, y, n, col):
        ax.text(x, y, n, color="white", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=10,
                clip_on=False, bbox=dict(boxstyle="circle,pad=0.24", facecolor=col, edgecolor="white", linewidth=1.0))

    badge(q1 + 0.36, v1 - shift / 2, "1", IR_C)    # beside its arrow, between the two curves
    badge(q2, v_c + 0.24, "2", BR_C)               # above the charge curve, over its arrow
    badge(0.25, 4.16, "3", WN_C)                   # upper band
    badge(4.30, 2.72, "3", WN_C)                   # lower band


    # right: one row per choice, label column then bar column
    labels = fig.add_axes([0.418, 0.200, 0.330, 0.600])
    a2 = fig.add_axes([0.772, 0.200, 0.160, 0.600])
    a2.patch.set_visible(False)
    y = np.arange(3)[::-1]
    for yy, (choice, method, mode, col, v) in zip(y, rows):
        a2.barh(yy, v, 0.42, facecolor=col, alpha=0.30, edgecolor=col, lw=1.1, zorder=3)
        a2.text(max(v, 0) + 3, yy, f"{v:+.0f}%".replace("-", "\u2212"), fontsize=9.0, color=col, va="center",
                ha="left", fontweight="bold", zorder=6)
        labels.text(0, yy + 0.17, method, fontsize=7.2, color=col, ha="left", va="center", fontweight="bold")
        labels.text(0, yy - 0.19, mode, fontsize=7.2, color="#3B4450", ha="left", va="center")
        labels.text(-0.075, yy, choice[0], fontsize=7.6, color="white", fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.26", facecolor=col, edgecolor="white", linewidth=1.0))
    for axis in (labels, a2):
        axis.set_ylim(-0.55, 2.55)
        axis.set_yticks([])
    labels.set_xlim(0, 1)
    labels.axis("off")
    a2.axvline(0, color="#3B4450", lw=1.2, zorder=5)
    a2.set_xlim(-50, 50)
    a2.set_xticks([-50, 0])
    a2.set_xlabel("change in fitted loss\nvs recommended DMA (%)", fontsize=6.8, labelpad=1.5)
    a2.tick_params(axis="x", labelsize=7.2, length=2.2, pad=1.5)
    a2.spines[["top", "right", "left"]].set_visible(False)

    fig.text(0.075, 0.935, "Same data, three analysis choices", fontsize=10.2, color=NAVY, fontweight="bold",
             va="center")
    fig.text(0.975, 0.935, "phantom LAM and LLI", fontsize=8.4, color=GREY, va="center", ha="right",
             style="italic")
    fig.text(0.975, 0.855, "LG M50T (1)  \u00b7  Molicel P45B (2, 3)", fontsize=6.4, color=GREY, va="center",
             ha="right")

    out = os.path.join(paths.OUTPUT, "figures", "Graphical_abstract.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print("wrote", out, [(r[0], round(r[4], 1)) for r in rows])


if __name__ == "__main__":
    main()
