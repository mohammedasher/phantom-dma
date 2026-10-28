"""SI figure: why the silicon signal is sharp on discharge and blurred on charge.

(a) Potential window over which each material changes state (10-90 % of its lithiation),
    from the reference curves used in the fit, on delithiation and lithiation.
(b, c) Share of the negative-electrode current carried by silicon and by graphite at
    equilibrium, from the composite relation SOC_NE(U) = phi*SOC_Si(U) + (1-phi)*SOC_Gr(U)
    used in the fit: s_Si(U) = phi*dSOC_Si/dU / dSOC_NE/dU. Mapped onto full-cell SOC with
    the fitted beginning-of-life parameters of the same cell (alignment only).
(d, e) Measured differential voltage at beginning of life with the transition point delta.
P45B, 2.94-4.2 V; discharge in (b, d), charge in (c, e).

Needs: reference_curves/ and results/P45B_discharge_2.94-4.2, results/P45B_charge_2.94-4.2.
Run:   python figures/figS2_current_partition.py
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
from matplotlib.patches import Rectangle
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Colour by material (validated pair), ink and recessive tones.
SI, GR = "#C98A2E", "#0091D4"
INK, MUTED, FAINT, GRID = "#1F2A36", "#5B6470", "#EEF1F4", "#E3E7EB"
NAVY = "#003E74"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8, "axes.edgecolor": "#3B4450",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
REF = os.path.join(paths.HERE, "reference_curves")
BRANCHES = {"discharge": ("graphite_delithiation.csv", "SiO_delithiation.csv", "P45B_discharge_2.94-4.2", "delithiation"),
            "charge": ("graphite_lithiation.csv", "SiO_lithiation.csv", "P45B_charge_2.94-4.2", "lithiation")}
PHI_RANGE = (0.186, 0.212)   # fitted silicon share at beginning of life, P45B configurations (Table S2)
U_MAX = 0.8                  # V vs Li/Li+, upper end of the potential grid
SMOOTH_MV = 10.0             # Gaussian smoothing of dSOC/dU in potential, mV (sampling noise of the curves)
Z_MAX = 0.40                 # full-cell SOC range shown in (d-g)
Z_HIDE_CHARGE = 0.04         # charge share not drawn below this z_FC (noise at the start of the Si curve)


def lithiation_of_potential(fname):
    """Lithiation fraction as a monotone (non-increasing) function of potential."""
    d = pd.read_csv(os.path.join(REF, fname)).sort_values("OCV")
    d = d.groupby("OCV", as_index=False).z.mean()
    z = np.minimum.accumulate(np.clip(d.z.values, 0, 1))
    return PchipInterpolator(d.OCV.values, z, extrapolate=False), d.OCV.min()


def partition(branch, phi, n=6000):
    fg, g0 = lithiation_of_potential(BRANCHES[branch][0])
    fs, s0 = lithiation_of_potential(BRANCHES[branch][1])
    u = np.linspace(max(g0, s0) + 1e-4, U_MAX, n)
    zg, zs = np.nan_to_num(fg(u), nan=0.0), np.nan_to_num(fs(u), nan=0.0)
    sig = SMOOTH_MV / 1000 / (u[1] - u[0])
    dsi = -phi * gaussian_filter1d(np.gradient(zs, u), sig)
    dgr = -(1 - phi) * gaussian_filter1d(np.gradient(zg, u), sig)
    tot = dsi + dgr
    share = np.where(tot > 0, dsi / np.where(tot > 0, tot, 1), np.nan)
    return u, zg, zs, phi * zs + (1 - phi) * zg, share, tot


def window(fname, lo=0.1, hi=0.9):
    """Potentials (V) at which the material is 90 % and 10 % lithiated."""
    f, u0 = lithiation_of_potential(fname)
    u = np.linspace(u0 + 1e-4, 2.0, 20000)
    z = np.nan_to_num(f(u), nan=0.0)
    return u[np.argmin(abs(z - hi))], u[np.argmin(abs(z - lo))]


def share_on_cell_soc(branch, theta, phi):
    """Silicon share of the current against full-cell SOC, using z_NE = sigma_NE + nu_NE * z_FC."""
    u, zg, zs, zne, s, tot = partition(branch, phi)
    zfc = (zne - theta[2]) / theta[0]
    o = np.argsort(zfc)
    grid = np.linspace(0, Z_MAX, 801)
    return grid, np.clip(np.interp(grid, zfc[o], np.nan_to_num(s[o], nan=0.0)), 0, 1)


def style(ax):
    ax.tick_params(labelsize=6.5, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=paths.RESULTS)
    p.add_argument("--cell", default="cell1")
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "FigS2_current_partition.png"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fig = plt.figure(figsize=(7.2, 7.6), dpi=300)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.62, 1.0, 1.0], hspace=0.62, wspace=0.18,
                          left=0.14, right=0.98, top=0.96, bottom=0.06)

    # ---- (a, b) reference potentials, with the 10-90 % span of each material shaded
    for col, branch in enumerate(("discharge", "charge")):
        axo = fig.add_subplot(gs[0, col])
        for fname, colour, name in ((BRANCHES[branch][0], GR, "graphite"), (BRANCHES[branch][1], SI, "silicon")):
            d = pd.read_csv(os.path.join(REF, fname))
            lo_u, hi_u = window(fname)
            axo.axvspan(lo_u, min(hi_u, U_MAX), color=colour, alpha=0.13, lw=0)
            axo.plot(d.OCV, d.z, color=colour, lw=1.5)
            for zz, uu in ((0.9, lo_u), (0.1, hi_u)):
                if uu <= U_MAX:
                    axo.plot(uu, zz, "o", ms=3.2, color=colour, mec="white", mew=0.6, zorder=5)
        lab = {"discharge": ((0.15, 0.55, "graphite"), (0.52, 0.42, "silicon")),
               "charge": ((0.215, 0.80, "graphite"), (0.33, 0.17, "silicon"))}[branch]
        for x, y, name in lab:
            axo.text(x, y, name, color=GR if name == "graphite" else SI, fontsize=6.5, fontweight="bold")
        axo.set_xlim(0, U_MAX)
        axo.set_ylim(0, 1.0)
        axo.set_xlabel(r"potential vs Li/Li$^{+}$ (V)", fontsize=7)
        if col == 0:
            axo.set_ylabel("lithiation fraction", fontsize=7)
        else:
            axo.set_yticklabels([])
        axo.grid(color=GRID, lw=0.6)
        axo.set_axisbelow(True)
        style(axo)
        axo.set_title(f"({'ab'[col]})  reference curves, {BRANCHES[branch][3]}", fontsize=7.4, color=INK,
                      loc="left", fontweight="bold")

    # ---- (a) potential windows: the infographic strip
    ax = fig.add_subplot(gs[1, :])
    info = {}
    for branch, y in (("discharge", 1.0), ("charge", 0.0)):
        g, s = BRANCHES[branch][0], BRANCHES[branch][1]
        (g_lo, g_hi), (s_lo, s_hi) = window(g), window(s)
        info[branch] = (g_lo, g_hi, s_lo, s_hi)
        h = 0.26
        ax.add_patch(Rectangle((g_lo, y + 0.02), g_hi - g_lo, h, color=GR, lw=0))
        ax.add_patch(Rectangle((s_lo, y - h - 0.02), min(s_hi, U_MAX) - s_lo, h, color=SI, lw=0))
        ov_lo, ov_hi = max(g_lo, s_lo), min(g_hi, s_hi)
        if ov_hi > ov_lo:
            ax.add_patch(Rectangle((ov_lo, y - h - 0.06), ov_hi - ov_lo, 2 * h + 0.12, fill=False,
                                   hatch="////", ec=INK, lw=0.6))
            ax.text(ov_hi + 0.008, y + 0.15, "both change state", ha="left", va="center", fontsize=6.3, color=INK)
        else:
            ax.annotate("", xy=(s_lo, y), xytext=(g_hi, y), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
            ax.text((g_hi + s_lo) / 2, y + h + 0.1, "no overlap", ha="center", fontsize=6.3, color=INK)
        ax.text(-0.012, y, f"{BRANCHES[branch][3]}\n(cell {branch})", ha="right", va="center", fontsize=6.6,
                color=INK)
        if branch == "charge":
            ax.text(g_lo - 0.005, y + 0.15, "graphite", color=GR, fontsize=6.2, va="center", ha="right",
                    fontweight="bold")
            ax.text(s_hi - 0.005, y - 0.15, "silicon", color="white", fontsize=6.2, va="center", ha="right",
                    fontweight="bold")
    ax.text(info["discharge"][0] + 0.005, 1.15, "graphite", color="white", fontsize=6.2, va="center", fontweight="bold")
    ax.text(info["discharge"][2] + 0.005, 0.85, "silicon", color="white", fontsize=6.2, va="center", fontweight="bold")
    ax.text(U_MAX - 0.005, 0.85, f"to {info['discharge'][3]:.2f} V →", color="white", fontsize=6.0, va="center",
            ha="right")
    ax.set_xlim(0, U_MAX)
    ax.set_ylim(-0.45, 1.5)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2.5)
    ax.set_xlabel(r"potential vs Li/Li$^{+}$ (V); bars span 10–90 % of each material's lithiation", fontsize=7)
    ax.text(-0.155, 1.0, "(c)", transform=ax.transAxes, fontsize=8, color=INK, fontweight="bold")

    # ---- (b, c) who carries the current; (d, e) what the measured curve shows
    for col, branch in enumerate(("discharge", "charge")):
        run = BRANCHES[branch][2]
        f = np.load(os.path.join(a.results, run, "fits", f"{a.cell}_RPT_000.npz"))
        theta, z, uexp, delta = f["theta"], f["z"], f["U_exp"], float(f["delta"])

        axs = fig.add_subplot(gs[2, col])
        grid, sh = share_on_cell_soc(branch, theta, theta[4])
        # the first points of the silicon lithiation curve are noisy: not drawn below z_FC = 0.04 on charge
        show = grid >= (Z_HIDE_CHARGE if branch == "charge" else 0.0)
        axs.fill_between(grid, 0, 100 * sh, where=show, color=SI, lw=0)
        axs.fill_between(grid, 100 * sh, 100, where=show, color=GR, lw=0, alpha=0.85)
        cross = grid[np.where((sh[:-1] >= 0.5) & (sh[1:] < 0.5))[0]]
        handover = cross[0] if len(cross) else None
        axs.text(0.10 if branch == "discharge" else 0.07, 35, "silicon", fontsize=6.5, color="white",
                 fontweight="bold", ha="center")
        axs.text(0.97, 0.88, "graphite", transform=axs.transAxes, fontsize=6.5, color="white", ha="right",
                 fontweight="bold")
        msg = "steep fall in\nsilicon share" if branch == "discharge" else "gradual fall in\nsilicon share"
        axs.text(0.97, 0.50, msg, transform=axs.transAxes, fontsize=5.9, color="white", ha="right", va="top")
        axs.set_xlim(0, Z_MAX)
        axs.set_ylim(0, 100)
        axs.set_title(f"({'de'[col]})  cell {branch}: {BRANCHES[branch][3]} of the NE", fontsize=7.4, color=INK,
                      loc="left", fontweight="bold")
        if col == 0:
            axs.set_ylabel("share of NE current (%)", fontsize=7)
        else:
            axs.set_yticklabels([])
        style(axs)

        axd = fig.add_subplot(gs[3, col], sharex=axs)
        g = np.gradient(savgol_filter(uexp, 11, 2), z)
        sel = z <= Z_MAX
        axd.axvspan(0, delta, color=FAINT, lw=0, zorder=0)
        axd.plot(z[sel], g[sel], color=NAVY, lw=1.2, zorder=3)
        axd.axvline(delta, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        axd.text(delta + 0.006, 3.7, rf"$\delta$ = {delta:.3f}", fontsize=6.2, color=MUTED)
        axd.text(delta / 2, 0.25, "voltage\nterm", fontsize=5.6, color=MUTED, ha="center")
        if handover is not None:
            for axx in (axs, axd):
                axx.axvline(handover, color=INK, lw=0.7, zorder=4)
            note = "50 % silicon share:\nsharp peak" if branch == "discharge" else \
                "50 % silicon share:\nno peak"
            axd.annotate(note, xy=(handover, 2.75), xytext=(handover + (0.012 if branch == "discharge" else 0.045), 2.75),
                         fontsize=6.0, color=INK, va="center",
                         arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=0))
        axd.set_ylim(0, 4)
        axd.set_xlabel(r"full-cell SOC, $z_{\mathrm{FC}}$", fontsize=7)
        if col == 0:
            axd.set_ylabel(r"measured d$V$/d$z_{\mathrm{FC}}$ (V)", fontsize=7)
        else:
            axd.set_yticklabels([])
        axd.grid(axis="y", color=GRID, lw=0.6)
        axd.set_axisbelow(True)
        style(axd)
        axd.set_title(f"({'fg'[col]})  measured, beginning of life", fontsize=7.4, color=INK, loc="left",
                      fontweight="bold")
        print(f"{branch}: delta {delta:.3f}; 50 % silicon share at z_FC {handover}; windows (V) {np.round(info[branch], 3)}")

    fig.savefig(a.out, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
