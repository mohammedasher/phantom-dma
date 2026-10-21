"""Fig. 1: one fit shown in four ways (P45B cell1, C/20 discharge, RPT 0, 2.5-4.2 V).

Panels: cell voltage, dV/dz, dz/dV, and the electrode potentials on the full-cell
SOC axis. The transition point delta is marked; the voltage term is fitted on one
side of it and dV/dz on the other.

Needs: results/P45B_discharge_2.5-4.2 and the P45B data (paths.py).
Run:   python figures/fig1_fit_workflow.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from phantom_dma.analyses import HALFCELLS
from phantom_dma.data import p45b_curve
from phantom_dma.model import CellModel, read_halfcell

CELL, RPT, RUN = "cell1", 0, "P45B_discharge_2.5-4.2"
NAVY, SKY, GREY, GREYL = "#003E74", "#8CC4E8", "#5B6470", "#A8B2BC"
RED, GREEN, AMBER = "#C0392B", "#2E8B57", "#C2570A"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.9, "axes.edgecolor": "#3B4450"})


def binned(z, v, n_bins=150):
    """Bin the raw curve: mean voltage and the scatter about a smooth trend in each bin."""
    order = np.argsort(z)
    z, v = z[order], v[order]
    trend = savgol_filter(v, min(501, len(v) - (1 - len(v) % 2)), 3)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    mean, sd = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 3:
            mean[b], sd[b] = v[sel].mean(), (v - trend)[sel].std()
    centres = (edges[:-1] + edges[1:]) / 2
    ok = np.isfinite(mean)
    return centres[ok], mean[ok], sd[ok]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=paths.RESULTS)
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "Fig1_fit_workflow.png"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fit = np.load(os.path.join(a.results, RUN, "fits", f"{CELL}_RPT_{RPT:03d}.npz"))
    theta, z_fc, delta = fit["theta"], fit["z"], float(fit["delta"])
    pe, gr, si = HALFCELLS[("P45B", "discharge")]
    model = CellModel(pe, gr, si)
    u_pe, u_ne, v_mod = model.voltage(z_fc, theta)
    z_ne = theta[0] * z_fc + theta[2]

    def own_curve(name):
        df = read_halfcell(name).sort_values("z")
        return PchipInterpolator(df["z"].values, df["OCV"].values, extrapolate=True)

    u_gr, u_si = own_curve(gr)(z_ne), own_curve(si)(z_ne)

    raw = p45b_curve(CELL, RPT, "discharge")
    raw = raw[(raw.V >= 2.5) & (raw.V <= 4.2)]
    z_raw = (raw.SOC - raw.SOC.min()) / (raw.SOC.max() - raw.SOC.min())
    zc, v_mean, v_sd = binned(z_raw.values, raw.V.values)

    def deriv(y, x):
        return np.gradient(savgol_filter(y, 11, 2) if len(y) > 12 else y, x)

    dv_e, dv_m = deriv(v_mean, zc), deriv(v_mod, z_fc)
    s_dv = np.r_[np.hypot(v_sd[0], v_sd[1]) / (zc[1] - zc[0]),
                 np.hypot(v_sd[:-2], v_sd[2:]) / (zc[2:] - zc[:-2]),
                 np.hypot(v_sd[-2], v_sd[-1]) / (zc[-1] - zc[-2])]
    rmse = 1000 * np.sqrt(np.mean((fit["U_exp"] - v_mod) ** 2))

    fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.5), dpi=300)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.945, bottom=0.095, wspace=0.28, hspace=0.32)
    v_side = (0, delta)  # discharge: voltage term below delta, dV/dz term above

    def mark(axis, label=False):
        axis.axvspan(*v_side, color=SKY, alpha=0.28, lw=0, zorder=0)
        axis.axvline(delta, color=GREY, lw=0.8, ls="--", zorder=1)
        if label:
            axis.text(delta + 0.01, 0.96, r"$\delta$", transform=axis.get_xaxis_transform(),
                      fontsize=7.5, color=GREY, va="top")

    legend = [Patch(facecolor=GREYL, alpha=0.35, label="measured (±1σ)"),
              Line2D([0], [0], color="#6B7480", lw=1.0, label="measured (mean)"),
              Line2D([0], [0], color=NAVY, lw=1.0, ls="--", label="model"),
              Patch(facecolor=SKY, alpha=0.28, label="voltage-fit region")]

    a0 = ax[0, 0]; mark(a0, True)
    a0.fill_between(zc, v_mean - v_sd, v_mean + v_sd, color=GREYL, alpha=0.35, lw=0)
    a0.plot(zc, v_mean, color="#6B7480", lw=1.0)
    a0.plot(z_fc, v_mod, color=NAVY, lw=1.0, ls="--")
    a0.set_ylabel("cell voltage (V)", fontsize=8.4)
    a0.legend(handles=legend, fontsize=6.2, frameon=False, loc="lower right", labelspacing=0.3)
    a0.text(0.97, 0.40, f"RMSE {rmse:.1f} mV", transform=a0.transAxes, fontsize=7.0, color=GREY, ha="right")

    a1 = ax[0, 1]; mark(a1, True)
    a1.fill_between(zc, dv_e - s_dv, dv_e + s_dv, color=GREYL, alpha=0.35, lw=0)
    a1.plot(zc, dv_e, color="#6B7480", lw=1.0)
    a1.plot(z_fc, dv_m, color=NAVY, lw=1.0, ls="--")
    a1.set_ylabel(r"$\mathrm{d}V/\mathrm{d}z_{\mathrm{FC}}$ (V)", fontsize=8.4)
    lo, hi = np.percentile(dv_e, [2, 98]); a1.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))

    a2 = ax[1, 0]; mark(a2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_e, ic_m, s_ic = 1 / dv_e, 1 / dv_m, np.abs(s_dv) / dv_e ** 2
    a2.fill_between(zc, ic_e - s_ic, ic_e + s_ic, color=GREYL, alpha=0.35, lw=0)
    a2.plot(zc, ic_e, color="#6B7480", lw=1.0)
    a2.plot(z_fc, ic_m, color=NAVY, lw=1.0, ls="--")
    a2.set_ylabel(r"$\mathrm{d}z_{\mathrm{FC}}/\mathrm{d}V$ (V$^{-1}$)", fontsize=8.4)
    fin = np.isfinite(ic_e); lo, hi = np.percentile(ic_e[fin], [2, 98])
    a2.set_ylim(lo - 0.15 * abs(hi - lo), hi + 0.15 * abs(hi - lo))

    a3 = ax[1, 1]
    a3.plot(z_fc, u_pe, color=RED, lw=1.8, label="positive")
    a3.plot(z_fc, u_ne, color=GREEN, lw=1.8, label="negative (blend)")
    a3.plot(z_fc, u_gr, color="#8791A0", lw=1.0, ls="--", label="graphite")
    a3.plot(z_fc, u_si, color=AMBER, lw=1.0, ls=":", label="silicon")
    a3.set_ylabel("electrode potential (V)", fontsize=8.4)
    a3.legend(fontsize=6.6, frameon=False, loc="center right", labelspacing=0.25)
    a3.text(0.97, 0.70, rf"$\phi_{{\mathrm{{Si}}}}$ = {theta[4]:.3f}", transform=a3.transAxes,
            fontsize=7.0, color=GREY, ha="right")

    for k, axis in enumerate(ax.ravel()):
        axis.set_xlim(0, 1)
        axis.tick_params(labelsize=7.2, length=2.4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#EEF1F4", lw=0.7); axis.set_axisbelow(True)
        axis.text(0.012, 1.015, f"({'abcd'[k]})", transform=axis.transAxes, fontsize=7.8,
                  color="#3B4450", fontweight="bold", va="bottom")
        if k >= 2:
            axis.set_xlabel("full-cell state of charge", fontsize=8.4)
    fig.savefig(a.out, dpi=300, facecolor="white")
    print(f"wrote {a.out}: cell {CELL}, RPT {RPT}, RMSE {rmse:.2f} mV, phi_Si {theta[4]:.3f}, delta {delta:.3f}")


if __name__ == "__main__":
    main()
