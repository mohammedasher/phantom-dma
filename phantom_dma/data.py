"""Read the measured pOCV curves and pulse resistances from the raw data files.

Every curve is returned as a table (pandas DataFrame) with the columns
    V    cell voltage (V)
    SOC  full-cell state of charge: 1 = full, 0 = empty
    Q    charge passed since the start of the step (Ah)
"""
import os

import numpy as np
import pandas as pd

import paths

# RPTs available for each cell type.
P45B_RPTS = [0, 50, 100, 200, 300, 400, 500]      # number of ageing cycles completed
M50T_RPTS = list(range(16))                       # test index in Kirkaldy et al.
M50T_PULSE_RPTS = [0, 2, 4, 6, 8, 10, 12, 14]     # M50T tests that include the pulse (GITT) data

M50T_POCV_CURRENT_A = 0.5                         # C/10 for the LG M50T

# Step numbers of the C/20 charge and discharge in the P45B pOCV test
# (see P45B_step_index_map.csv in the data record).
P45B_CHARGE_STEP = 6
P45B_DISCHARGE_STEP = 8


def soc_from_charge(q, invert):
    """Scale charge to 0-1. Discharge curves are inverted so that SOC = 1 is full."""
    span = q.iloc[-1] - q.iloc[0]
    z = (q - q.iloc[0]) / span if span else q * 0.0
    return 1 - z if invert else z


# ------------------------------------------------------------------ Molicel P45B
def p45b_file(cell, rpt, test="pOCV-C20"):
    """Path of one P45B data file, for example P45B_25degC_cell1_RPT050_pOCV-C20.csv."""
    return os.path.join(paths.P45B_DATA, f"P45B_25degC_{cell}_RPT{rpt:03d}_{test}.csv")


def p45b_curve(cell, rpt, branch):
    """C/20 pOCV charge or discharge of one P45B cell at one RPT."""
    columns = ["Voltage / V", "Current / A", "Total Capacity / Ah", "Discharge Capacity / Ah", "Step Index"]
    df = pd.read_csv(p45b_file(cell, rpt), usecols=columns)
    if branch == "charge":
        # Constant-current part of the C/20 charge (the CV hold that follows is excluded
        # by the voltage window used in every fit).
        step = df[(df["Step Index"] == P45B_CHARGE_STEP) & (df["Current / A"] > 0)]
        q = step["Total Capacity / Ah"] - step["Total Capacity / Ah"].iloc[0]
        soc = soc_from_charge(step["Total Capacity / Ah"], invert=False)
    else:
        step = df[(df["Step Index"] == P45B_DISCHARGE_STEP) & (df["Current / A"] < 0)]
        q = step["Discharge Capacity / Ah"] - step["Discharge Capacity / Ah"].iloc[0]
        soc = soc_from_charge(step["Discharge Capacity / Ah"], invert=True)
    return pd.DataFrame({"V": step["Voltage / V"].values, "SOC": soc.values, "Q": q.values})


# ------------------------------------------------------------------ LG M50T
def m50t_file(cell_name, rpt, kind):
    """Path of one file of Kirkaldy et al. (experiment 5). cell_name is "D" or "E"."""
    if kind == "pOCV":
        folder, name = "0.1C Voltage Curves", f"Expt 5 - cell {cell_name} - RPT{rpt} - 0.1C discharge data.csv"
    else:
        folder, name = "GITT Voltage Curves", f"Expt 5 - cell {cell_name} - RPT{rpt} - 25-pulse GITT 0.5C discharge data.csv"
    return os.path.join(paths.M50T_DATA, "Processed Timeseries Data", folder, f"Cell {cell_name}", name)


def m50t_curve(cell_name, rpt, ir_corrected=False):
    """C/10 pOCV discharge of one LG M50T cell, optionally IR-corrected (see the paper)."""
    df = pd.read_csv(m50t_file(cell_name, rpt, "pOCV"), usecols=["Voltage (V)", "Charge (mA.h)"])
    q = df["Charge (mA.h)"] / 1000.0
    out = pd.DataFrame({"V": df["Voltage (V)"].values,
                        "SOC": soc_from_charge(q, invert=True).values,
                        "Q": q.values})
    # Rests repeat the same charge value; keep the first sample of each.
    out = out.drop_duplicates(subset="SOC", keep="first").reset_index(drop=True)
    if ir_corrected:
        soc_pulse, r_ohm = m50t_resistance(cell_name, rpt)
        if ir_corrected == "10s":
            # Sensitivity test (Section 4.4, Table S8): a larger correction, with the resistance
            # read 10 s into each C/2 pulse from rest instead of at 0.1 s. It includes part of
            # the slow, history-dependent resistance, which does not transfer to the low-rate trace.
            soc_r, ratio = m50t_resistance_ratio(cell_name, rpt, 10.0)
            r_ohm = r_ohm * np.interp(soc_pulse, soc_r, ratio)
        out["V"] = out["V"].values + M50T_POCV_CURRENT_A * np.interp(out["SOC"].values, soc_pulse, r_ohm)
    return out


def r_inst(v, i, k, threshold=0.002):
    """Instantaneous resistance (mOhm) of the pulse that starts at sample k.

    The rest voltage is the last sample before the voltage leaves its rest level
    by more than 2 mV; the loaded voltage is the first sample under current
    (0.1 s after the step, at 10 Hz logging).
    """
    base = v[max(k - 20, 0):k - 2].mean() if k > 4 else v[k - 1]
    j = k
    while j > 0 and abs(v[j - 1] - base) > threshold:
        j -= 1
    return abs(v[k] - v[j - 1]) / abs(i[k]) * 1000.0


def m50t_resistance_ratio(cell_name, rpt, t_long=10.0):
    """Ratio of the t_long (s) resistance to R_0.1s for each C/2 GITT pulse, SOC ascending.

    Both use the same rest voltage (as in r_inst); the long value is read t_long seconds
    after the current step. Returns (soc, ratio).
    """
    g = pd.read_csv(m50t_file(cell_name, rpt, "GITT"))
    t, v, i, q = g.iloc[:, 0].values, g.iloc[:, 1].values, g.iloc[:, 2].values / 1000.0, g.iloc[:, 3].values
    starts = np.where((i[1:] < -0.1) & (i[:-1] >= -0.1))[0] + 1
    soc, ratio = [], []
    for k in starts:
        base = v[max(k - 20, 0):k - 2].mean() if k > 4 else v[k - 1]
        j = k
        while j > 0 and abs(v[j - 1] - base) > 0.002:
            j -= 1
        rest = v[j - 1]
        m = min(np.searchsorted(t, t[k] + t_long), len(t) - 1)
        soc.append(1 - q[k - 1] / q.max())
        ratio.append((rest - v[m]) / (rest - v[k]))
    order = np.argsort(soc)
    return np.array(soc)[order], np.array(ratio)[order]


def m50t_resistance(cell_name, rpt):
    """R_0.1s (Ohm) against SOC from the 25-pulse C/2 GITT discharge, SOC ascending."""
    g = pd.read_csv(m50t_file(cell_name, rpt, "GITT"))
    v, i, q = g.iloc[:, 1].values, g.iloc[:, 2].values / 1000.0, g.iloc[:, 3].values
    starts = np.where((i[1:] < -0.1) & (i[:-1] >= -0.1))[0] + 1
    soc = np.array([1 - q[k - 1] / q.max() for k in starts])
    r = np.array([r_inst(v, i, k) for k in starts]) / 1000.0
    order = np.argsort(soc)
    return soc[order], r[order]


def m50t_throughput_kah(cell_name):
    """Cumulative charge throughput (kAh) at each RPT, from the summary file of Kirkaldy et al."""
    f = os.path.join(paths.M50T_DATA, "Summary Data", "Performance Summary",
                     f"Expt 5 - cell {cell_name} (25degC) - Processed Data.csv")
    t = pd.read_csv(f)
    return pd.Series(t["Charge Throughput [A h]"].values / 1000.0, index=t["Ageing Sets"].values)


# ------------------------------------------------------------------ one call for any analysis
def load_curves(settings, cell):
    """All curves {rpt: curve} for one analysis (a dict from analyses.ANALYSES) and one cell."""
    from .analyses import CELLS
    name = CELLS[settings["cell"]][cell]
    if settings["cell"] == "P45B":
        return {r: p45b_curve(name, r, settings["branch"]) for r in P45B_RPTS}
    corrected = settings.get("ir_corrected", False)
    rpts = M50T_PULSE_RPTS if corrected else M50T_RPTS
    return {r: m50t_curve(name, r, corrected) for r in rpts}
