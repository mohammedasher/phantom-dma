"""Fig. 3: what the IR correction does to an LG M50T pOCV curve.

The C/10 pOCV discharge at the first RPT is corrected by adding
I_pOCV x R_0.1s(SOC), with R_0.1s from the GITT pulses (first or last RPT).
(a) voltage against capacity, measured and corrected;
(b) |dV/dQ|: the mean IR term cancels, leaving a small residual (inset);
(c) |dQ/dV|: the peaks move by about I x R;
(d) the IR lift with the first- and the last-RPT resistance;
(e) the change in |dV/dQ| from resistance growth alone;
(f) the shift of each dQ/dV peak from resistance growth alone.
Panels (d)-(f) apply both resistances to the same first-RPT curve, so they
isolate the effect of resistance growth.

Uncertainty. Filled bands: repeating the measurement (per-pulse noise of R and
the second cell of the same experiment). Error bars in (f) and the numbers in
(a), (c) and (d) also cover the processing choices (interpolation between
pulses, extrapolation beyond them, peak location and smoothing) and, for peak
positions, the pOCV voltage noise (Monte Carlo).

Needs: the LG M50T C/10 and GITT files (paths.py).
Run:   python figures/fig3_ir_correction.py
Writes output/figures/Fig3_IR_correction.png and Fig3_key_numbers.csv next to it.
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks, savgol_filter

from phantom_dma.data import M50T_POCV_CURRENT_A, m50t_file, m50t_resistance_ratio
from phantom_dma.resistance import ir_lift, ir_lift_ensemble, m50t_pulse_capacities

CELL = "E"                 # cell2 of the LG M50T (Kirkaldy et al. name, used to open its files)
SECOND_CELL = "D"          # cell1, the other 25 degC cell, for the cell-to-cell part of the bands
FIRST_RPT, LAST_RPT = 0, 14
I_POCV = M50T_POCV_CURRENT_A                   # 0.5 A, C/10

SMOOTHING = (121, 61)                          # Savitzky-Golay windows: dV/dQ and dQ/dV
SMOOTHING_ALTERNATIVES = [(101, 41), (141, 81)]
SIGMA_V = 0.0543e-3        # V, scatter of the C/10 pOCV about a local cubic fit
N_MC = 160                 # Monte Carlo draws of that voltage noise, for the peak positions
PEAK_MATCH_V = 0.05        # peaks closer than this are taken to be the same peak

NAVY, POOL, SKY, BAND = "#003E74", "#0091D4", "#8CC4E8", "#EEF6FC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8, "text.color": "k",
                     "axes.labelcolor": "k", "xtick.color": "k", "ytick.color": "k"})


# ------------------------------------------------------------------ data
def read_pocv():
    """C/10 discharge of the first RPT: capacity (Ah), voltage (V), relative SOC, mean current (A)."""
    d = pd.read_csv(m50t_file(CELL, FIRST_RPT, "pOCV"))
    d.columns = ["t", "V", "I", "Q", "T"]
    d = d[d.I < -100]                          # samples under current (mA)
    q = d.Q.abs().values / 1000
    return q, d.V.values, 1 - q / q.max(), -d.I.mean() / 1000


# ------------------------------------------------------------------ differential curves
class Curves:
    """dV/dQ and dQ/dV of curves sampled on the capacity grid q, with given smoothing windows."""

    def __init__(self, q, v_grid, windows=SMOOTHING):
        self.q, self.v_grid = q, v_grid
        self.dq = q[1] - q[0]
        self.w_dvdq, self.w_dqdv = windows

    def smooth(self, y):
        return savgol_filter(y, self.w_dvdq, 3)

    def dvdq(self, v):
        return self.smooth(np.gradient(self.smooth(v), self.dq))

    def dqdv(self, v):
        vs = self.smooth(v)
        order = np.argsort(vs)
        q_on_v = np.interp(self.v_grid, vs[order], self.q[order])
        return savgol_filter(np.gradient(q_on_v, self.v_grid[1] - self.v_grid[0]), self.w_dqdv, 3)


def peak_voltages(dqdv, v_grid, refine=False):
    """Voltages of the dQ/dV peaks. refine=True fits a parabola through the three
    samples around each peak, removing the 0.77 mV step of the voltage grid."""
    z = -dqdv
    idx, _ = find_peaks(z, prominence=0.5)
    if not refine:
        return v_grid[idx]
    step = v_grid[1] - v_grid[0]
    out = []
    for i in idx:
        if 0 < i < len(z) - 1:
            den = z[i - 1] - 2 * z[i] + z[i + 1]
            out.append(v_grid[i] + (np.clip(0.5 * (z[i - 1] - z[i + 1]) / den, -1, 1) * step if den else 0.0))
        else:
            out.append(v_grid[i])
    return np.array(out)


def window_peak(dqdv, v_grid, refine=False):
    """Voltage of the highest dQ/dV point between 3.40 and 3.50 V (the peak in the (c) inset)."""
    z = -dqdv
    sl = np.where((v_grid > 3.40) & (v_grid < 3.50))[0]
    i = sl[np.argmax(z[sl])]
    if refine and 0 < i < len(z) - 1:
        den = z[i - 1] - 2 * z[i] + z[i + 1]
        step = v_grid[1] - v_grid[0]
        return v_grid[i] + (np.clip(0.5 * (z[i - 1] - z[i + 1]) / den, -1, 1) * step if den else 0.0)
    return v_grid[i]


def matched_shift(p_from, p_to, p_ref):
    """Shift (mV) of the peak nearest p_ref, from list p_from to list p_to; None if not found."""
    i = np.argmin(abs(p_from - p_ref))
    if abs(p_from[i] - p_ref) < PEAK_MATCH_V:
        j = np.argmin(abs(p_to - p_from[i]))
        if abs(p_to[j] - p_from[i]) < PEAK_MATCH_V:
            return 1e3 * (p_to[j] - p_from[i])
    return None


# ------------------------------------------------------------------ the analysis
def analyse():
    q_raw, v_raw, soc, i_measured = read_pocv()
    q_total = q_raw.max()
    q = np.linspace(0.02, q_total - 0.02, 3000)
    v_grid = np.linspace(3.0, 4.15, 1500)
    curves = Curves(q, v_grid)
    on_q = lambda y: np.interp(q, q_raw, y)

    v_meas = on_q(v_raw)
    lift_first = on_q(ir_lift(soc, CELL, FIRST_RPT, I_POCV))
    lift_last = on_q(ir_lift(soc, CELL, LAST_RPT, I_POCV))
    v_corr, v_corr_last = v_meas + lift_first, v_meas + lift_last

    r = dict(q=q, q_total=q_total, v_grid=v_grid, v_meas=v_meas, v_corr=v_corr,
             lift_first=lift_first, lift_last=lift_last, i_measured=i_measured,
             dvdq_meas=curves.dvdq(v_meas), dvdq_corr=curves.dvdq(v_corr), dvdq_last=curves.dvdq(v_corr_last),
             dqdv_meas=curves.dqdv(v_meas), dqdv_corr=curves.dqdv(v_corr),
             r_mean=lift_first.mean() / I_POCV)

    # Alternative lifts. "random": repeat measurement (noise, second cell). "choice": processing.
    cells = (CELL, SECOND_CELL)
    rnd_first, cho_first = [np.array([on_q(m) for m in e]) for e in ir_lift_ensemble(soc, FIRST_RPT, I_POCV, cells)]
    rnd_last, cho_last = [np.array([on_q(m) for m in e]) for e in ir_lift_ensemble(soc, LAST_RPT, I_POCV, cells)]
    pairs_random = list(zip(rnd_first, rnd_last))
    pairs_all = pairs_random + list(zip(cho_first, cho_last))
    r.update(rnd_first=rnd_first, rnd_last=rnd_last, cho_first=cho_first, cho_last=cho_last)

    # Peak shifts from resistance growth: same curve corrected with the first- and the last-RPT R.
    p_first = peak_voltages(curves.dqdv(v_meas + lift_first), v_grid)
    p_last = peak_voltages(curves.dqdv(v_meas + lift_last), v_grid)
    shifts = []
    for p in p_first:
        j = np.argmin(abs(p_last - p))
        if abs(p_last[j] - p) < PEAK_MATCH_V:
            shifts.append((p, 1e3 * (p_last[j] - p)))

    # Spread of each shift over the alternative lifts, smoothings and peak locators.
    spread = {p: [] for p, _ in shifts}

    def collect(curves_, a, b, refine):
        pa = peak_voltages(curves_.dqdv(v_meas + a), v_grid, refine)
        pb = peak_voltages(curves_.dqdv(v_meas + b), v_grid, refine)
        for p, _ in shifts:
            s = matched_shift(pa, pb, p)
            if s is not None:
                spread[p].append(s)

    alternative_curves = [Curves(q, v_grid, w) for w in SMOOTHING_ALTERNATIVES]
    for refine in (False, True):
        for a, b in pairs_all:
            collect(curves, a, b, refine)
        for c in alternative_curves:
            collect(c, lift_first, lift_last, refine)

    # Monte Carlo of the pOCV voltage noise. Differences taken on the same measured
    # curve do not see it (corrected - measured = lift exactly); peak positions do.
    rng = np.random.default_rng(0)
    mc = {p: [] for p, _ in shifts}
    mc_window = []
    for _ in range(N_MC):
        v_noisy = np.interp(q, q_raw, v_raw + rng.normal(0, SIGMA_V, len(v_raw)))
        d_first, d_last = curves.dqdv(v_noisy + lift_first), curves.dqdv(v_noisy + lift_last)
        pa, pb = peak_voltages(d_first, v_grid), peak_voltages(d_last, v_grid)
        for p, _ in shifts:
            s = matched_shift(pa, pb, p)
            if s is not None:
                mc[p].append(s)
        mc_window.append(1e3 * (window_peak(d_first, v_grid) - window_peak(curves.dqdv(v_noisy), v_grid)))
    r["mc_sd"] = {p: (np.std(v, ddof=1) if len(v) > 2 else 0.0) for p, v in mc.items()}
    r["mc_seen"] = {p: len(v) for p, v in mc.items()}
    r["mc_window_sd"] = float(np.std(mc_window, ddof=1))
    r["shifts"], r["shift_spread"] = shifts, spread

    # Peak shift produced by the correction itself (inset of panel c).
    def correction_shift(lift, curves_=curves, refine=False):
        return 1e3 * (window_peak(curves_.dqdv(v_meas + lift), v_grid, refine)
                      - window_peak(curves_.dqdv(v_meas), v_grid, refine))
    r["window_shift"] = correction_shift(lift_first)
    r["window_shift_all"] = ([correction_shift(a, refine=rf) for a, _ in pairs_all for rf in (False, True)]
                             + [correction_shift(lift_first, c, rf) for c in alternative_curves for rf in (False, True)])

    # Lift at Q = 2.2 Ah (panel a inset), and growth of the mean lift (panel d).
    k = np.argmin(abs(q - 2.2))
    r["lift_22"] = [1e3 * lift_first[k],
                    1e3 * min(rnd_first.min(0)[k], cho_first.min(0)[k]),
                    1e3 * max(rnd_first.max(0)[k], cho_first.max(0)[k])]
    r["growth_all"] = [1e3 * (b - a).mean() for a, b in pairs_all]

    # dV/dQ bands: corrected minus measured (b), and last minus first RPT (e).
    r["band_b"] = np.array([-1e3 * (curves.dvdq(v_meas + a) - r["dvdq_meas"]) for a, _ in pairs_random])
    r["band_e"] = np.array([-1e3 * (curves.dvdq(v_meas + b) - curves.dvdq(v_meas + a)) for a, b in pairs_random])

    r["pulses_first"] = m50t_pulse_capacities(soc, q_raw, CELL, FIRST_RPT)
    r["pulses_last"] = m50t_pulse_capacities(soc, q_raw, CELL, LAST_RPT)
    return r


# ------------------------------------------------------------------ the figure
def title(a, letter, txt):
    a.set_title(f"({letter})  {txt}", loc="left", fontsize=9, fontweight="bold", pad=7)


def clean(a):
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)


def make_figure(r, out):
    q, q_total, v_grid = r["q"], r["q_total"], r["v_grid"]
    v_meas, v_corr = r["v_meas"], r["v_corr"]
    lift_first, lift_last = r["lift_first"], r["lift_last"]
    middle = (q > 0.3) & (q < q_total - 0.3)       # dV/dQ is shown away from the ends
    low_soc = 0.75 * q_total                       # shaded: the last quarter of the capacity

    fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.6), gridspec_kw={"hspace": 0.5, "wspace": 0.33, "top": 0.95,
                                                                   "bottom": 0.12, "left": 0.07, "right": 0.985})

    def shade(a):
        a.axvspan(low_soc, q_total, color=BAND, zorder=0, lw=0)

    # (a) V-Q, with an inset at 2.2 Ah
    a = ax[0, 0]
    shade(a)
    a.plot(q, v_meas, c=SKY, ls="--", lw=1.4, label="measured pOCV")
    a.plot(q, v_corr, c=NAVY, lw=1.5, label="IR-corrected")
    a.set_xlim(0, q_total)
    a.set_ylim(2.5, 4.25)
    a.set_xlabel("Discharge capacity (Ah)")
    a.set_ylabel("Voltage (V)")
    a.legend(frameon=False, fontsize=7.4, loc="lower left")
    ia = inset_axes(a, width="42%", height="40%", loc="upper right", borderpad=0.8)
    sl = (q > 2.12) & (q < 2.28)
    ia.plot(q[sl], v_meas[sl], c=SKY, ls="--", lw=1.4)
    ia.plot(q[sl], v_corr[sl], c=NAVY, lw=1.5)
    ia.set_xticks([])
    ia.set_yticks([])
    i = np.argmin(abs(q - 2.2))
    ia.annotate("", xy=(2.2, v_corr[i]), xytext=(2.2, v_meas[i]),
                arrowprops=dict(arrowstyle="<->", lw=1.0, shrinkA=0, shrinkB=0))
    nom, lo, hi = r["lift_22"]
    ia.text(2.205, (v_corr[i] + v_meas[i]) / 2, f"{nom:.1f} ± {max(nom - lo, hi - nom):.2f} mV", fontsize=7.4,
            fontweight="bold", va="center", ha="left", bbox=dict(fc="white", ec="none", pad=0.8))
    ia.set_ylim(min(v_meas[sl].min(), v_corr[sl].min()) - 0.004, max(v_meas[sl].max(), v_corr[sl].max()) + 0.004)
    title(a, "a", "V–Q: shifted")
    clean(a)

    # (b) |dV/dQ|, with the residual (corrected - measured) in an inset
    a = ax[0, 1]
    shade(a)
    a.plot(q[middle], -r["dvdq_meas"][middle], c=SKY, ls="--", lw=1.4)
    a.plot(q[middle], -r["dvdq_corr"][middle], c=NAVY, lw=1.3)
    a.set_xlim(0, q_total)
    a.set_ylim(-0.03, 0.34)
    a.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    a.set_xlabel("Discharge capacity (Ah)")
    a.set_ylabel(r"$|\mathrm{d}V/\mathrm{d}Q|$ (V Ah$^{-1}$)")
    ib = inset_axes(a, width="44%", height="28%", loc="lower right", borderpad=0.9)
    ib.axvspan(low_soc, q_total, color=BAND, lw=0)
    ib.fill_between(q[middle], r["band_b"].min(0)[middle], r["band_b"].max(0)[middle], color=NAVY, alpha=0.28, lw=0)
    ib.plot(q[middle], -1e3 * (r["dvdq_corr"] - r["dvdq_meas"])[middle], c=NAVY, lw=1.0)
    ib.axhline(0, c="k", lw=0.5, ls=":")
    ib.set_xlim(0, q_total)
    ib.set_ylim(-4, 4)
    ib.set_xticks([])
    ib.tick_params(labelsize=6.5)
    ib.yaxis.tick_right()
    ib.set_yticks([-3, 0, 3])
    ib.text(0.03, 0.93, r"corrected $-$ measured", transform=ib.transAxes, fontsize=6.5, va="top")
    ib.text(0.03, 0.07, r"mV Ah$^{-1}$", transform=ib.transAxes, fontsize=6.3, va="bottom", color="0.3")
    title(a, "b", r"d$V$/d$Q$: $I\bar{R}$ cancels, small residual")
    clean(a)

    # (c) |dQ/dV|, with the 3.40-3.50 V peak in an inset
    a = ax[0, 2]
    d_meas, d_corr = r["dqdv_meas"], r["dqdv_corr"]
    a.plot(v_grid, -d_meas, c=SKY, ls="--", lw=1.4)
    a.plot(v_grid, -d_corr, c=NAVY, lw=1.4)
    a.set_xlim(3.3, 4.15)
    a.set_ylim(0, 13)
    a.set_xlabel("Voltage (V)")
    a.set_ylabel(r"$|\mathrm{d}Q/\mathrm{d}V|$ (Ah V$^{-1}$)")
    ic = inset_axes(a, width="38%", height="40%", loc="upper left", borderpad=0.9)
    sl = (v_grid > 3.40) & (v_grid < 3.50)
    ic.plot(v_grid[sl], -d_meas[sl], c=SKY, ls="--", lw=1.2)
    ic.plot(v_grid[sl], -d_corr[sl], c=NAVY, lw=1.3)
    ic.set_xticks([])
    ic.set_yticks([])
    p_meas, p_corr = v_grid[sl][np.argmax(-d_meas[sl])], v_grid[sl][np.argmax(-d_corr[sl])]
    yy = (-d_corr[sl]).max() * 1.02
    ic.annotate("", xy=(p_corr, yy), xytext=(p_meas, yy), arrowprops=dict(arrowstyle="->", lw=0.8))
    nom, alls = r["window_shift"], r["window_shift_all"]
    err = max(abs(nom - min(alls)), abs(max(alls) - nom)) + 2 * r["mc_window_sd"]
    ic.text((p_meas + p_corr) / 2, yy * 1.05, f"+{nom:.0f} ± {err:.1f} mV", ha="center", fontsize=6.6)
    ic.set_ylim(None, yy * 1.20)
    title(a, "c", r"d$Q$/d$V$: peaks shift by $I\bar{R}$")
    clean(a)

    # (d) the IR lift with the first- and last-RPT resistance
    a = ax[1, 0]
    shade(a)
    lift_mean_mv = r["r_mean"] * I_POCV * 1e3
    a.fill_between(q, lift_mean_mv, lift_first * 1e3, color=SKY, alpha=0.40, lw=0)
    a.fill_between(q, r["rnd_first"].min(0) * 1e3, r["rnd_first"].max(0) * 1e3, color=NAVY, alpha=0.30, lw=0,
                   label="uncertainty in $R$")
    a.fill_between(q, r["rnd_last"].min(0) * 1e3, r["rnd_last"].max(0) * 1e3, color=POOL, alpha=0.30, lw=0)
    a.axhline(lift_mean_mv, c=NAVY, ls="--", lw=1.1)
    a.plot(q, lift_first * 1e3, c=NAVY, lw=1.6, label="first RPT")
    a.plot(q, lift_last * 1e3, c=POOL, lw=1.6, label="last RPT")
    a.plot(r["pulses_first"], np.interp(r["pulses_first"], q, lift_first) * 1e3, "o", ms=2.5, c=NAVY, zorder=6,
           label="measured pulses")
    a.plot(r["pulses_last"], np.interp(r["pulses_last"], q, lift_last) * 1e3, "o", ms=2.5, c=POOL, zorder=6)
    a.set_xlim(0, q_total)
    a.set_ylim(12.5, 18)
    a.set_xlabel("Discharge capacity (Ah)")
    a.set_ylabel(r"IR lift  $\Delta V_{\mathrm{IR}} = I\,R(Q)$ (mV)")
    a.annotate(r"$I\bar{R}$", xy=(2.6, lift_mean_mv), xytext=(2.6, 14.35), fontsize=8.5, ha="center",
               arrowprops=dict(arrowstyle="-", lw=0.6))
    a.annotate(r"$I\,[R(Q)-\bar{R}]$", xy=(4.3, 13.95), xytext=(3.1, 12.75), fontsize=8.2, ha="center",
               va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6))
    a.legend(frameon=False, fontsize=6.6, loc="upper left", labelspacing=0.3)
    k = np.argmin(abs(q - 1.0))
    a.annotate("", xy=(1.0, lift_last[k] * 1e3), xytext=(1.0, lift_first[k] * 1e3),
               arrowprops=dict(arrowstyle="->", lw=0.9))
    growth = 1e3 * np.mean(lift_last - lift_first)
    g = r["growth_all"]
    a.text(1.06, (lift_first[k] + lift_last[k]) * 500,
           f"+{growth:.1f} $^{{+{max(g) - growth:.1f}}}_{{-{growth - min(g):.1f}}}$ mV", fontsize=7.4, va="center")
    title(a, "d", "IR lift at first vs last RPT")
    clean(a)

    # (e) change in |dV/dQ| from resistance growth
    a = ax[1, 1]
    shade(a)
    a.fill_between(q[middle], r["band_e"].min(0)[middle], r["band_e"].max(0)[middle], color=POOL, alpha=0.35, lw=0,
                   label="uncertainty in $R$")
    a.plot(q[middle], -1e3 * (r["dvdq_last"] - r["dvdq_corr"])[middle], c=POOL, lw=1.3)
    a.axhline(0, c="k", lw=0.5, ls=":")
    a.set_xlim(0, q_total)
    a.set_ylim(-5, 5)
    a.legend(frameon=False, fontsize=6.8, loc="upper left", labelspacing=0.3)
    a.set_xlabel("Discharge capacity (Ah)")
    a.set_ylabel(r"$\Delta|\mathrm{d}V/\mathrm{d}Q|$ (mV Ah$^{-1}$)")
    title(a, "e", r"d$V$/d$Q$ change from $R$ growth")
    clean(a)

    # (f) dQ/dV peak shifts from resistance growth
    a = ax[1, 2]
    for k, (p, v) in enumerate(r["shifts"]):
        lo, hi = shift_range(r, p, v)
        a.bar(k, v, width=0.45, color=POOL)
        a.errorbar(k, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor="k", elinewidth=1.0,
                   capsize=3, capthick=1.0)
        a.text(k, min(hi + 0.18, 5.35), f"+{v:.1f}", ha="center", fontsize=7.6)
    n = len(r["shifts"])
    a.set_xticks(range(n))
    a.set_xticklabels([f"{p:.2f} V" for p, _ in r["shifts"]], fontsize=7.6)
    a.set_xlim(-0.6, n - 0.4)
    a.set_ylim(-0.7, 5.7)
    a.axhline(0, c="k", lw=0.6)
    a.set_xlabel("d$Q$/d$V$ peak")
    a.set_ylabel("Phantom peak shift (mV)")
    title(a, "f", r"d$Q$/d$V$ peak shift from $R$ growth")
    clean(a)

    fig.text(0.07, 0.035, r"At C/10 (0.5 A):  1 m$\Omega$ $\rightarrow$ 0.5 mV shift in V–Q and d$Q$/d$V$ peaks;  "
             r"0 in d$V$/d$Q$.     Faster pOCV scales all shifts:  C/5 $\times$2  $\cdot$  C/2 $\times$5  $\cdot$  "
             r"1C $\times$10.", fontsize=7.8, ha="left", va="bottom")
    ratios = [m50t_resistance_ratio(c, FIRST_RPT) for c in (CELL, SECOND_CELL)]
    mid = np.mean([x[(s > 0.3) & (s < 0.7)].mean() for s, x in ratios])
    low = max(x.max() for s, x in ratios)
    fig.text(0.07, 0.008, r"Instantaneous (0.1 s) resistance only, i.e. a lower bound:  at beginning of life the 10 s "
             rf"resistance is {mid:.1f}$\times$ larger at mid-SOC and up to {low:.1f}$\times$ at low SOC.",
             fontsize=7.8, ha="left", va="bottom")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)


def shift_range(r, p, v):
    """Low and high end of a peak shift: spread over the alternatives, widened by 2 SD of the voltage noise."""
    e = r["shift_spread"][p]
    lo, hi = (min(e), max(e)) if e else (v, v)
    return lo - 2 * r["mc_sd"][p], hi + 2 * r["mc_sd"][p]


def key_numbers(r):
    lift_all_first = list(r["rnd_first"]) + list(r["cho_first"])
    lift_all_last = list(r["rnd_last"]) + list(r["cho_last"])
    rows = [dict(quantity="pOCV current: measured vs assumed 0.5 A (%)",
                 nominal=100 * (r["i_measured"] - I_POCV) / I_POCV, low=np.nan, high=np.nan),
            dict(quantity="IR lift at Q = 2.2 Ah (mV)", nominal=r["lift_22"][0], low=r["lift_22"][1], high=r["lift_22"][2]),
            dict(quantity="mean IR lift, first RPT (mV)", nominal=1e3 * r["lift_first"].mean(),
                 low=1e3 * min(m.mean() for m in lift_all_first), high=1e3 * max(m.mean() for m in lift_all_first)),
            dict(quantity="mean IR lift, last RPT (mV)", nominal=1e3 * r["lift_last"].mean(),
                 low=1e3 * min(m.mean() for m in lift_all_last), high=1e3 * max(m.mean() for m in lift_all_last)),
            dict(quantity="IR lift growth, first -> last RPT (mV)",
                 nominal=1e3 * np.mean(r["lift_last"] - r["lift_first"]),
                 low=min(r["growth_all"]), high=max(r["growth_all"])),
            dict(quantity="dQ/dV peak shift from the correction, 3.4-3.5 V window (mV)", nominal=r["window_shift"],
                 low=min(r["window_shift_all"]), high=max(r["window_shift_all"]))]
    for p, v in r["shifts"]:
        lo, hi = shift_range(r, p, v)
        rows.append(dict(quantity=f"phantom peak shift from R growth at {p:.2f} V (mV)", nominal=v, low=lo, high=hi,
                         sd_from_pOCV_voltage_noise=r["mc_sd"][p], detected_in_MC=f"{r['mc_seen'][p]}/{N_MC}"))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(paths.OUTPUT, "figures", "Fig3_IR_correction.png"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    r = analyse()
    make_figure(r, a.out)
    table = key_numbers(r)
    table.round(3).to_csv(os.path.join(os.path.dirname(a.out), "Fig3_key_numbers.csv"), index=False)
    print(f"wrote {a.out}")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
