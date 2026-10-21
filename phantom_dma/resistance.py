"""Pulse resistance R_0.1s and the IR correction, for Figs. 2 and 3.

R_0.1s is the voltage step between the last rest sample and the first sample
under current (0.1 s after the current step, at 10 Hz logging), divided by the
pulse current. The same definition, data.r_inst, is used for both cell types.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d

from .data import m50t_file, m50t_resistance, p45b_file, r_inst

# Per-pulse measurement noise of R_0.1s (Ohm), from the scatter of the rest
# samples before each pulse. The same value holds for both cell types.
R_NOISE_OHM = 0.03e-3

# A discharge pulse counts as truncated if the voltage reaches this level
# (the 2.5 V cut-off, plus 5 mV).
CUTOFF_V = 2.505


# ------------------------------------------------------------------ LG M50T
def m50t_pulses(cell_name, rpt):
    """Every discharge pulse of the 25-pulse GITT test of one LG M50T cell.

    Returns a dict with
        Q      capacity discharged over the whole GITT test (Ah)
        s0     SOC at the start of each pulse (%)
        s1     SOC at the end of each pulse (%)
        R      R_0.1s of each pulse (mOhm)
        full   False for pulses cut short by the 2.5 V limit
    """
    d = pd.read_csv(m50t_file(cell_name, rpt, "GITT"))
    v, i, q = d.iloc[:, 1].values, d.iloc[:, 2].values / 1000.0, d.iloc[:, 3].values   # current in mA in the file
    starts = np.where((i[1:] < -0.1) & (i[:-1] >= -0.1))[0] + 1
    stops = np.where((i[1:] >= -0.1) & (i[:-1] < -0.1))[0] + 1
    ends = np.array([stops[stops > k][0] for k in starts])
    q_total = q.max()
    return dict(Q=q_total / 1000.0,
                s0=np.array([1 - q[k - 1] / q_total for k in starts]) * 100,
                s1=np.array([1 - q[k - 1] / q_total for k in ends]) * 100,
                R=np.array([r_inst(v, i, k) for k in starts]),
                full=np.array([v[k:e].min() > CUTOFF_V for k, e in zip(starts, ends)]))


def m50t_pulse_capacities(soc, q, cell_name, rpt):
    """Where on a pOCV curve (capacity axis) each GITT pulse was measured.

    soc and q are the relative SOC and discharged capacity of the pOCV samples.
    """
    s, _ = m50t_resistance(cell_name, rpt)
    order = np.argsort(soc)
    return np.interp(s, np.asarray(soc)[order], np.asarray(q)[order])


def resistance_at(x, s, r, interp="linear", extrap="flat"):
    """R at the SOC values x, from pulses measured at SOC s (ascending).

    interp: "linear", "pchip" or "nearest" between pulses.
    extrap: "flat" (hold the end values) or "linear" beyond the first and last pulse.
    """
    x = np.asarray(x, float)
    inside = np.clip(x, s[0], s[-1])
    if interp == "linear":
        y = np.interp(inside, s, r)
    elif interp == "pchip":
        y = PchipInterpolator(s, r, extrapolate=False)(inside)
    else:
        y = interp1d(s, r, kind="nearest", bounds_error=False, fill_value=(r[0], r[-1]))(inside)
    y = np.asarray(y, float)
    y = np.where(np.isnan(y), np.interp(inside, s, r), y)
    if extrap == "linear":
        below, above = x < s[0], x > s[-1]
        if below.any():
            y[below] = r[0] + (r[1] - r[0]) / (s[1] - s[0]) * (x[below] - s[0])
        if above.any():
            y[above] = r[-1] + (r[-1] - r[-2]) / (s[-1] - s[-2]) * (x[above] - s[-1])
    return y


def ir_lift(soc, cell_name, rpt, current_a, interp="linear", extrap="flat", dr=0.0):
    """IR correction (V) at each pOCV sample: current x R_0.1s(SOC).

    The pulse resistance is mapped onto the pOCV by relative SOC. dr shifts
    every pulse resistance by a fixed amount (Ohm), to test measurement noise.
    """
    s, r = m50t_resistance(cell_name, rpt)
    return current_a * resistance_at(soc, s, r + dr, interp, extrap)


def ir_lift_ensemble(soc, rpt, current_a, cell_names=("E", "D")):
    """Alternative IR corrections, to show how uncertain the correction is.

    random:  what would change if the measurement were repeated. Both 25 degC
             cells, each with every pulse resistance shifted by -noise, 0, +noise.
    choice:  what changes if the same data are processed differently. The
             first cell, with linear / PCHIP / nearest interpolation between
             pulses and flat / linear extrapolation beyond them.
    Both are returned as arrays of lift (V), one row per member.
    """
    random = [ir_lift(soc, c, rpt, current_a, dr=d)
              for c in cell_names for d in (-R_NOISE_OHM, 0.0, R_NOISE_OHM)]
    choice = [ir_lift(soc, cell_names[0], rpt, current_a, interp=i, extrap=e)
              for i in ("linear", "pchip", "nearest") for e in ("flat", "linear")]
    return np.array(random), np.array(choice)


# ------------------------------------------------------------------ Molicel P45B
def p45b_cc_step(cell, rpt, step):
    """Constant-current samples of one step of the C/20 pOCV test of a P45B cell.

    Samples whose current differs from the step's median current by more than
    1 % (the CV part and the transitions) are dropped. Column dQ is the charge
    (Ah) passed since the previous kept sample.
    """
    d = pd.read_csv(p45b_file(cell, rpt, "pOCV-C20"))
    b = d[d["Step Index"] == step].copy()
    current = b["Current / A"].median()
    b = b[(b["Current / A"] - current).abs() < 0.01 * abs(current)]
    b["dQ"] = b["Current / A"].abs() * b["Test Time / s"].diff().fillna(0) / 3600
    return b


def p45b_pulse_resistance(cell, rpt):
    """R_0.1s (mOhm) of the 1C, 30 s discharge pulses of one P45B cell, in test order
    (70, 50 and 30 % SOC)."""
    d = pd.read_csv(p45b_file(cell, rpt, "rate-and-pulse"))
    t, v, i = d["Test Time / s"].values, d["Voltage / V"].values, d["Current / A"].values
    starts = np.where((i[1:] < -3) & (i[:-1] >= -3))[0] + 1
    r = []
    for k in starts:
        e = k
        while e < len(i) and i[e] < -3:
            e += 1
        if 25 <= t[e - 1] - t[k] <= 35:          # keep the 30 s pulses only
            r.append(r_inst(v, i, k))
    return np.array(r)
