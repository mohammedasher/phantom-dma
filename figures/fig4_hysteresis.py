"""Fig. 4: charge-discharge voltage gap of the Molicel P45B and how it changes with ageing.

Both C/20 pOCV branches are put on one capacity axis: the discharge is
reflected and the charge shifted so that both reach the same capacity at
4.2 V. The gap is dV(Q) = V_charge(Q) - V_discharge(Q) on a 1 mAh grid,
smoothed with a 0.20 Ah centred moving average.
(a) charge and discharge at the first RPT, one cell, and how the gap is read;
(b) the gap along the capacity axis at the first and the last RPT;
(c) the change in the gap at the same capacity;
(d) the last-RPT gap moved along the capacity axis to best match the first
    (least RMS difference): the features move rather than shrink.
Panels (b)-(d) show the mean of the two cells; the bands give their range.

The script also prints how the numbers change with the smoothing window and
with matching the branches at the low-voltage end instead of at 4.2 V.

Needs: the P45B data (paths.py).
Run:   python figures/fig4_hysteresis.py
Writes output/figures/Fig4_hysteresis.png and Fig4_key_numbers.csv next to it.
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
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from phantom_dma.data import P45B_CHARGE_STEP, P45B_DISCHARGE_STEP
from phantom_dma.resistance import p45b_cc_step

CELLS = ["cell1", "cell2"]
RPTS = {"First": 0, "Last": 500}
PLOT_CELL = "cell2"                  # the cell drawn in panel (a)
PLOT_CELL_LABEL = "one cell"         # legend text for panel (a)
WINDOWS = [100, 200, 300]            # smoothing windows in samples of 1 mAh: 0.10 / 0.20 / 0.30 Ah
WINDOW = 200                         # the one used
SHIFT_GRID = np.arange(0, 0.401, 0.005)   # capacity shifts tried in (d), Ah
TRIM_AFTER_SHIFT = 0.10              # Ah dropped at the start of the moved curve
MARKED_Q = [0.35, 1.0, 2.5]          # capacities of the numbered gaps in (a) and (b)

NAVY, POOL, SKY = "#003E74", "#0091D4", "#8CC4E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8,
                     "text.color": "k", "axes.labelcolor": "k"})


# ------------------------------------------------------------------ data
def branches(cell, rpt, match="top"):
    """Charge and discharge (constant-current samples) on one capacity axis, column Qt.

    match="top": both branches reach the same capacity at 4.2 V (used in the paper).
    match="bottom": both start at the same capacity at the low-voltage end.
    """
    ch = p45b_cc_step(cell, rpt, P45B_CHARGE_STEP)
    dc = p45b_cc_step(cell, rpt, P45B_DISCHARGE_STEP)
    ch["Q"], dc["Q"] = ch["dQ"].cumsum(), dc["dQ"].cumsum()
    q_dc, q_ch = dc.Q.max(), ch.Q.max()
    dc["Qt"] = q_dc - dc.Q
    ch["Qt"] = ch.Q + (q_dc - q_ch) if match == "top" else ch.Q
    return ch.sort_values("Qt"), dc.sort_values("Qt")


def gap(ch, dc, window):
    """Smoothed gap (mV) on a 1 mAh grid; the ends lost to the moving average are dropped."""
    lo, hi = max(ch.Qt.min(), dc.Qt.min()), min(ch.Qt.max(), dc.Qt.max())
    g = np.arange(lo, hi, 0.001)
    raw = (np.interp(g, ch.Qt, ch["Voltage / V"]) - np.interp(g, dc.Qt, dc["Voltage / V"])) * 1e3
    y = pd.Series(raw).rolling(window, center=True).mean().values
    keep = ~np.isnan(y)
    return g[keep], y[keep]


def nan_mean(arrays):
    """Mean over cells, ignoring cells with no value at that point."""
    a = np.array(arrays, float)
    n = np.isfinite(a).sum(0)
    return np.where(n > 0, np.nansum(a, 0) / np.maximum(n, 1), np.nan)


def best_shift(G, first, last):
    """Capacity shift of the last-RPT gap that best matches the first-RPT gap (least RMS)."""
    rms = []
    for s in SHIFT_GRID:
        moved = np.interp(G, G + s, last, left=np.nan, right=np.nan)
        ok = ~np.isnan(moved) & ~np.isnan(first)
        rms.append(np.sqrt(np.mean((first[ok] - moved[ok]) ** 2)))
    rms = np.array(rms)
    return SHIFT_GRID[int(np.nanargmin(rms))], rms


def narrowing_fraction(G, first, last, frac=0.75):
    """Fraction of the capacity window that holds `frac` of the total narrowing,
    on that curve's own finite range."""
    m = np.isfinite(first) & np.isfinite(last)
    g = G[m]
    net = np.cumsum(first[m] - last[m])
    net = net / net[-1]
    return (g[np.argmax(net >= frac)] - g[0]) / (g[-1] - g[0])


# ------------------------------------------------------------------ the analysis
def analyse():
    curves = {(c, s): branches(c, rpt) for c in CELLS for s, rpt in RPTS.items()}
    raw = {(c, s, w): gap(*curves[(c, s)], w) for c in CELLS for s in RPTS for w in WINDOWS}

    # One capacity window common to both cells and both RPTs.
    lo = max(g.min() for (c, s, w), (g, y) in raw.items() if w == WINDOW)
    hi = min(g.max() for (c, s, w), (g, y) in raw.items() if w == WINDOW)
    G = np.arange(lo, hi, 0.001)
    on_G = {k: np.interp(G, g, y, left=np.nan, right=np.nan) for k, (g, y) in raw.items()}

    r = dict(curves=curves, G=G)
    r["F"] = nan_mean([on_G[(c, "First", WINDOW)] for c in CELLS])           # two-cell mean, first RPT
    r["L"] = nan_mean([on_G[(c, "Last", WINDOW)] for c in CELLS])            # two-cell mean, last RPT
    r["Fc"] = np.array([on_G[(c, "First", WINDOW)] for c in CELLS])          # each cell
    r["Lc"] = np.array([on_G[(c, "Last", WINDOW)] for c in CELLS])
    r["Fk"] = np.array([nan_mean([on_G[(c, "First", w)] for c in CELLS]) for w in WINDOWS])   # each window
    r["Lk"] = np.array([nan_mean([on_G[(c, "Last", w)] for c in CELLS]) for w in WINDOWS])
    F, L, Fc, Lc, Fk, Lk = (r[k] for k in ("F", "L", "Fc", "Lc", "Fk", "Lk"))

    # Shift that best matches the two gaps. Its range covers each cell, each smoothing
    # window and every shift whose RMS is within 1 mV of the best.
    sh, rms = best_shift(G, F, L)
    sh_cells = [best_shift(G, Fc[i], Lc[i])[0] for i in range(len(CELLS))]
    sh_windows = [best_shift(G, Fk[i], Lk[i])[0] for i in range(len(WINDOWS))]
    flat = SHIFT_GRID[rms <= np.nanmin(rms) + 1.0]
    r.update(shift=sh, shift_cells=sh_cells, shift_windows=sh_windows, flat=flat,
             shift_lo=min(sh_cells + sh_windows + [flat.min()]),
             shift_hi=max(sh_cells + sh_windows + [flat.max()]))

    moved = np.interp(G, G + sh, L, left=np.nan, right=np.nan)
    moved[G < G[0] + sh + TRIM_AFTER_SHIFT] = np.nan
    ok = ~np.isnan(moved)
    r["rms_before"] = np.sqrt(np.mean((F[ok] - L[ok]) ** 2))
    r["rms_after"] = np.sqrt(np.mean((F[ok] - moved[ok]) ** 2))

    r["q75"] = narrowing_fraction(G, F, L)
    r["q75_all"] = ([narrowing_fraction(G, Fc[i], Lc[i]) for i in range(len(CELLS))]
                    + [narrowing_fraction(G, Fk[i], Lk[i]) for i in range(len(WINDOWS))])
    r["change_same_q"] = 100 * (L[ok].mean() / F[ok].mean() - 1)
    r["change_moved"] = 100 * (moved[ok].mean() / F[ok].mean() - 1)
    r["change_cells"] = [100 * (Lc[i][ok].mean() / Fc[i][ok].mean() - 1) for i in range(len(CELLS))]

    # Voltage noise left in the smoothed gap (scatter about a 51-point running mean).
    r["noise"] = np.nanstd(F - pd.Series(F).rolling(51, center=True).mean().values)
    r["bottom"] = bottom_matched()
    return r


def bottom_matched():
    """Main numbers again with the branches matched at the low-voltage end instead of 4.2 V."""
    raw = {(c, s): gap(*branches(c, rpt, "bottom"), WINDOW) for c in CELLS for s, rpt in RPTS.items()}
    lo = max(g.min() for g, y in raw.values())
    hi = min(g.max() for g, y in raw.values())
    G = np.arange(lo, hi, 0.001)
    on_G = {k: np.interp(G, g, y) for k, (g, y) in raw.items()}
    first = np.mean([on_G[(c, "First")] for c in CELLS], 0)
    last = np.mean([on_G[(c, "Last")] for c in CELLS], 0)
    rms = []
    for s in SHIFT_GRID:
        moved = np.interp(G, G + s, last, left=np.nan, right=np.nan)
        o = ~np.isnan(moved)
        rms.append(np.sqrt(np.mean((first[o] - moved[o]) ** 2)))
    sh = SHIFT_GRID[int(np.argmin(rms))]
    moved = np.interp(G, G + sh, last, left=np.nan, right=np.nan)
    moved[G < G[0] + sh + TRIM_AFTER_SHIFT] = np.nan
    o = ~np.isnan(moved)
    return dict(first=first.mean(), last=last.mean(), change=100 * (last[o].mean() / first[o].mean() - 1),
                shift=sh, change_moved=100 * (moved[o].mean() / first[o].mean() - 1))


# ------------------------------------------------------------------ the figure
def title(a, t):
    a.set_title(t, loc="left", fontsize=9, fontweight="bold", pad=7)


def clean(a):
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)


def number(a, x, y, k, dx=0, dy=0):
    """White number in a black circle."""
    a.text(x + dx, y + dy, str(k), ha="center", va="center", fontsize=7, fontweight="bold", color="white",
           bbox=dict(boxstyle="circle,pad=0.18", fc="k", ec="none"))


def log_gap_axis(a, xmax):
    a.set_yscale("log")
    a.set_yticks([20, 50, 100, 200, 400])
    a.set_yticklabels(["20", "50", "100", "200", "400"])
    a.yaxis.set_minor_formatter(NullFormatter())   # label the major ticks only
    a.set_ylim(15, 520)
    a.set_xlim(0, xmax)
    a.set_xlabel("Capacity (Ah)")
    a.set_ylabel(r"Voltage gap $\Delta V(Q)$ (mV)")


def make_figure(r, out):
    G, F, L, Fc, Lc, sh = r["G"], r["F"], r["L"], r["Fc"], r["Lc"], r["shift"]
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.0), gridspec_kw={"hspace": 0.45, "wspace": 0.28, "left": 0.08,
                                                                  "right": 0.98, "top": 0.94, "bottom": 0.08})

    # (a) the loop, with numbered gap arrows and an inset at the third one
    a = ax[0, 0]
    ch, dc = r["curves"][(PLOT_CELL, "First")]
    v_ch = lambda x: np.interp(x, ch.Qt, ch["Voltage / V"])
    v_dc = lambda x: np.interp(x, dc.Qt, dc["Voltage / V"])
    gg = np.linspace(max(ch.Qt.min(), dc.Qt.min()), min(ch.Qt.max(), dc.Qt.max()), 800)
    a.fill_between(gg, v_dc(gg), v_ch(gg), color=NAVY, alpha=0.12, lw=0)
    a.plot(ch.Qt, ch["Voltage / V"], c=NAVY, ls="--", lw=1.3, label="charge")
    a.plot(dc.Qt, dc["Voltage / V"], c=NAVY, lw=1.4, label="discharge")
    for k, q0 in enumerate(MARKED_Q, 1):
        v1, v2 = v_dc(q0), v_ch(q0)
        if k < 3:
            a.annotate("", xy=(q0, v2), xytext=(q0, v1), arrowprops=dict(arrowstyle="<->", lw=0.9, shrinkA=0, shrinkB=0))
            number(a, q0, v2, k, dy=0.1)
        else:
            a.plot([q0 - 0.12, q0 + 0.12, q0 + 0.12, q0 - 0.12, q0 - 0.12],
                   [v1 - 0.04, v1 - 0.04, v2 + 0.04, v2 + 0.04, v1 - 0.04], c="k", lw=0.6)
            number(a, q0, v2, k, dy=0.14)
    ia = inset_axes(a, width="34%", height="34%", loc="lower right", borderpad=1.0)
    q3 = MARKED_Q[2]
    sl = (gg > q3 - 0.12) & (gg < q3 + 0.12)
    ia.fill_between(gg[sl], v_dc(gg[sl]), v_ch(gg[sl]), color=NAVY, alpha=0.12, lw=0)
    ia.plot(gg[sl], v_ch(gg[sl]), c=NAVY, ls="--", lw=1.2)
    ia.plot(gg[sl], v_dc(gg[sl]), c=NAVY, lw=1.3)
    v1, v2 = v_dc(q3), v_ch(q3)
    ia.annotate("", xy=(q3, v2), xytext=(q3, v1), arrowprops=dict(arrowstyle="<->", lw=0.9, shrinkA=0, shrinkB=0))
    ia.text(q3 + 0.012, (v1 + v2) / 2, f"{1e3 * (v2 - v1):.0f} mV", fontsize=7, va="center")
    ia.set_xticks([])
    ia.set_yticks([])
    number(ia, q3 - 0.085, v2, 3)
    a.text(0.5, 2.72, r"gap at capacity $Q$:" + "\n" + r"$\Delta V(Q) = V_{\mathrm{ch}}(Q) - V_{\mathrm{dch}}(Q)$",
           fontsize=8, ha="left", va="center")
    a.set_xlim(0, 4.6)
    a.set_ylim(2.5, 4.35)
    a.set_xlabel("Capacity (Ah)")
    a.set_ylabel("Voltage (V)")
    a.legend(frameon=False, fontsize=7.4, loc="upper left", title=f"first RPT (C/20, {PLOT_CELL_LABEL})",
             title_fontsize=7.4, alignment="left")
    title(a, "(a)  Charge and discharge curves")
    clean(a)

    # (b) the gap along the capacity axis
    a = ax[0, 1]
    a.fill_between(G, F, L, color=POOL, alpha=0.30, lw=0, label="difference, shown in (c)")
    a.fill_between(G, Fc.min(0), Fc.max(0), color=NAVY, alpha=0.28, lw=0, label="cell-to-cell")
    a.fill_between(G, Lc.min(0), Lc.max(0), color=POOL, alpha=0.40, lw=0)
    a.plot(G, F, c=NAVY, lw=1.6, label="first RPT")
    a.plot(G, L, c=POOL, lw=1.6, label="last RPT")
    for k, q0 in enumerate(MARKED_Q, 1):
        y = np.interp(q0, G, F)
        a.plot(q0, y, "o", ms=3.5, c=NAVY, zorder=5)
        number(a, q0, y * 1.28, k)
    log_gap_axis(a, G[-1] + 0.05)
    a.legend(frameon=False, fontsize=6.8, loc="upper right", labelspacing=0.3)
    a.text(0.03, 0.04, f"each point = arrow length in (a)   |   mean gap {np.nanmean(F):.0f} → {np.nanmean(L):.0f} mV",
           transform=a.transAxes, fontsize=7.4, va="bottom")
    title(a, "(b)  Gap along the capacity axis")
    clean(a)

    # (c) change in the gap at the same capacity
    a = ax[1, 0]
    D, Dc = L - F, Lc - Fc
    q75 = G[0] + r["q75"] * (G[-1] - G[0])
    a.axhline(0, c="k", lw=0.6)
    a.fill_between(G, D, 0, where=D < 0, color=POOL, alpha=0.45, lw=0, label="gap narrowed")
    a.fill_between(G, D, 0, where=D > 0, color=SKY, alpha=0.45, lw=0, hatch="///", edgecolor=POOL, label="gap widened")
    a.fill_between(G, Dc.min(0), Dc.max(0), color=NAVY, alpha=0.22, lw=0, label="cell-to-cell")
    a.plot(G, D, c=NAVY, lw=1.2)
    a.axvline(q75, c="k", ls=":", lw=0.8)
    a.annotate(f"75% of the narrowing lies\nwithin the first {r['q75']:.0%} of capacity\n"
               f"(range {min(r['q75_all']):.0%}–{max(r['q75_all']):.0%})",
               xy=(q75, -30), xytext=(1.6, -62), fontsize=7.4, va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6))
    a.set_xlim(0, G[-1] + 0.05)
    a.set_ylim(-95, 30)
    a.set_xlabel("Capacity (Ah)")
    a.set_ylabel(r"Change in gap at same $Q$:" + "\nlast $-$ first (mV)")
    a.legend(frameon=False, fontsize=6.8, loc="upper right", labelspacing=0.3)
    title(a, "(c)  Change in gap = shaded area in (b)")
    clean(a)

    # (d) the last-RPT gap moved along the capacity axis
    a = ax[1, 1]
    L_moved = np.where(G + sh >= G[0] + sh + TRIM_AFTER_SHIFT, L, np.nan)
    a.plot(G, L, c=SKY, lw=1.1, label="last RPT")
    a.plot(G, F, c=NAVY, lw=1.6, label="first RPT")
    a.plot(G + sh, L_moved, c=POOL, lw=1.6, ls=":", label=f"last RPT moved +{sh:.2f} Ah")
    i = np.argmin(abs(G - 1.05))
    a.annotate("", xy=(G[i] + sh, L[i] * 1.2), xytext=(G[i], L[i] * 1.2), arrowprops=dict(arrowstyle="->", lw=0.9))
    err = max(sh - r["shift_lo"], r["shift_hi"] - sh)
    a.text(G[i] + sh / 2, L[i] * 1.38, f"+{sh:.2f} $\\pm$ {err:.2f} Ah", ha="center", fontsize=7.4)
    log_gap_axis(a, G[-1] + 0.05)
    a.legend(frameon=False, fontsize=7.2, loc="upper right")
    a.text(0.03, 0.04, f"change in mean gap (overlapping range):\n  at the same capacity   −{abs(r['change_same_q']):.0f}%\n"
           f"  after moving +{sh:.2f} Ah   −{abs(r['change_moved']):.0f}%", transform=a.transAxes, fontsize=7.4,
           va="bottom")
    title(a, "(d)  Features move rather than shrink")
    clean(a)

    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)


def key_numbers(r):
    Fc, Lc, b = r["Fc"], r["Lc"], r["bottom"]
    rows = [dict(quantity="mean gap, first RPT (mV)", nominal=np.nanmean(r["F"]),
                 low=min(np.nanmean(x) for x in Fc), high=max(np.nanmean(x) for x in Fc)),
            dict(quantity="mean gap, last RPT (mV)", nominal=np.nanmean(r["L"]),
                 low=min(np.nanmean(x) for x in Lc), high=max(np.nanmean(x) for x in Lc)),
            dict(quantity="capacity shift of the gap features (Ah)", nominal=r["shift"],
                 low=r["shift_lo"], high=r["shift_hi"]),
            dict(quantity="RMS mismatch before shift (mV)", nominal=r["rms_before"], low=np.nan, high=np.nan),
            dict(quantity="RMS mismatch after shift (mV)", nominal=r["rms_after"], low=np.nan, high=np.nan),
            dict(quantity="fraction of capacity holding 75% of the narrowing (%)", nominal=100 * r["q75"],
                 low=100 * min(r["q75_all"]), high=100 * max(r["q75_all"])),
            dict(quantity="change in mean gap at same capacity (%)", nominal=r["change_same_q"],
                 low=min(r["change_cells"]), high=max(r["change_cells"])),
            dict(quantity="change in mean gap after the shift (%)", nominal=r["change_moved"], low=np.nan, high=np.nan),
            dict(quantity="voltage noise left in the smoothed gap (mV)", nominal=r["noise"], low=np.nan, high=np.nan),
            dict(quantity="mean gap, first RPT, branches matched at low-V end instead (mV)", nominal=b["first"],
                 low=np.nan, high=np.nan),
            dict(quantity="narrowing, branches matched at low-V end instead (%)", nominal=b["change"],
                 low=np.nan, high=np.nan),
            dict(quantity="shift, branches matched at low-V end instead (Ah)", nominal=b["shift"],
                 low=np.nan, high=np.nan)]
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "Fig4_hysteresis.png"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    r = analyse()
    make_figure(r, a.out)
    table = key_numbers(r)
    table.round(3).to_csv(os.path.join(os.path.dirname(a.out), "Fig4_key_numbers.csv"), index=False)

    print(f"wrote {a.out}")
    print(f"common capacity window {r['G'][0]:.3f}-{r['G'][-1]:.3f} Ah")
    print("shift (Ah): each cell", [round(float(v), 3) for v in r["shift_cells"]],
          "| each smoothing window", [round(float(v), 3) for v in r["shift_windows"]],
          f"| within 1 mV of the best RMS {r['flat'].min():.3f}-{r['flat'].max():.3f}")
    b = r["bottom"]
    print(f"branches matched at the low-voltage end: mean gap {b['first']:.1f} -> {b['last']:.1f} mV, "
          f"after the shift {b['change_moved']:.1f} %")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
