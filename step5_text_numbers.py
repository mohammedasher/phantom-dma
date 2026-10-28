"""Step 5: the numbers quoted in the text of the paper that are not read directly off a
table or figure.

    python step5_text_numbers.py

Writes output/text_numbers.csv: where each number appears, what it is, the value
computed here, and the value as written in the paper. Run step 3 first (Table S8 holds the
uncertainty of the resistance bias). The data-based numbers need the raw data (see README).

Relative changes are 100 * (alt - ref) / ref of the two-cell mean, in % of the reference
value. Losses are in % of the beginning-of-life quantity.
"""
import os
import sys

import numpy as np
import pandas as pd

import paths
from phantom_dma import fit
from phantom_dma.data import (M50T_POCV_CURRENT_A, P45B_CHARGE_STEP, P45B_DISCHARGE_STEP, P45B_RPTS,
                              m50t_curve, m50t_resistance, m50t_resistance_ratio, p45b_curve, p45b_file, r_inst)
from phantom_dma.resistance import p45b_cc_step
from phantom_dma.uncertainty import paired_difference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))

RES = paths.RESULTS
M50T_AGEING = [2, 4, 6, 8, 10, 12, 14]
P45B_AGEING = [50, 100, 200, 300, 400, 500]
ROWS = []


def add(where, quantity, value, paper):
    ROWS.append(dict(where=where, quantity=quantity, computed=value, paper=paper))
    print(f"{where:<14} {quantity:<78} {value:<22} paper: {paper}")


def mean(analysis):
    return pd.read_csv(os.path.join(RES, analysis, "summary.csv")).groupby("RPT").mean(numeric_only=True)


def rel(ref, alt, key, rpts):
    r, a = mean(ref)[key], mean(alt)[key]
    return (100 * (a - r) / r).loc[list(rpts)]


def span(values, fmt="{:.1f}"):
    return fmt.format(min(values)) + " to " + fmt.format(max(values))


# ------------------------------------------------------------------ degradation modes
def resistance_bias():
    ref, alt = "M50T_discharge_IR-corrected", "M50T_discharge_uncorrected"
    x = rel(ref, alt, "LAM_PE_pct", M50T_AGEING)
    add("Abstract, 3.2", "LAM_PE overstated without IR correction, mean over ageing RPTs (%)", f"{x.mean():.1f}", "10")
    add("3.2", "  ... raised at every ageing RPT", str(bool((x > 0).all())), "every ageing RPT")
    for n, cell in enumerate(("cell1", "cell2")):
        a = pd.read_csv(os.path.join(RES, ref, f"{cell}_summary.csv")).set_index("RPT")
        b = pd.read_csv(os.path.join(RES, alt, f"{cell}_summary.csv")).set_index("RPT")
        c = (100 * (b.LAM_PE_pct - a.LAM_PE_pct) / a.LAM_PE_pct).loc[M50T_AGEING].mean()
        add("3.2", f"  ... {cell} alone (%)", f"{c:.1f}", ("15", "4")[n])
    d = (mean(alt).LLI_pct - mean(ref).LLI_pct).loc[M50T_AGEING].abs().max()
    add("3.2", "largest LLI change without IR correction (% of BoL)", f"{d:.2f}", "less than 0.2")
    with open(os.path.join(paths.OUTPUT, "tables", "TableS8_rows.tex")) as f:
        note = [l for l in f if l.startswith("% LAM_PE")][0].strip("% \n")
    add("3.2, Table S8", "LAM_PE overstatement averaged over ageing RPTs (% of BoL)", note,
        "0.70 +/- 0.45; 1.12 +/- 0.17; 0.28 +/- 0.15")

    ref, alt = "M50T_discharge_IR-corrected", "M50T_discharge_IR10s-corrected"
    pe, ne, si = (rel(ref, alt, k, M50T_AGEING) for k in ("LAM_PE_pct", "LAM_NE_pct", "LAM_Si_pct"))
    add("4.4", "10 s correction vs 0.1 s: LAM_PE, mean over ageing RPTs (%)", f"{pe.mean():.1f}", "-6.5")
    add("4.4", "  ... LAM_PE higher at any RPT", str(bool((pe > 0).any())), "no RPT")
    add("4.4", "  ... LAM_NE (%)", f"{ne.mean():.1f}", "+5")
    add("4.4", "  ... LAM_Si (%)", f"{si.mean():.1f}", "less than 3")
    s = {r: mean(r).LAM_Si_pct.loc[M50T_AGEING].mean() for r in
         ("M50T_discharge_uncorrected", "M50T_discharge_IR-corrected", "M50T_discharge_IR10s-corrected")}
    add("4.4", "  ... correction with the highest LAM_Si (mean over ageing RPTs)", max(s, key=s.get),
        "the instantaneous (0.1 s) correction")


def window_bias():
    ref, alt = "P45B_discharge_2.5-4.2", "P45B_discharge_3.0-4.2"
    late = [200, 300, 400, 500]
    for key, paper in (("LAM_Si_pct", "13 to 41"), ("LAM_NE_pct", "17 to 36"), ("LLI_pct", "6 to 8")):
        x = rel(ref, alt, key, late)
        add("Abstract, 3.3", f"3.0 V lower cut-off: {key} understated, RPT 200-500 (%)", span(-x, "{:.0f}"), paper)
    q = mean(ref).Q_exp_Ah, mean(alt).Q_exp_Ah
    fade = [100 * (1 - v.loc[500] / v.loc[0]) for v in q]
    add("3.3", "capacity fade to the last RPT, full window minus 3.0-4.2 V (% of BoL)", f"{fade[0] - fade[1]:.1f}",
        "about 2")
    x = rel(ref, "P45B_discharge_3.4-4.2_screen", "LAM_Si_pct", [500])
    add("3.3", "3.4 V lower cut-off: LAM_Si overstated at the last RPT (%)", f"{x.iloc[0]:.0f}", "two-thirds")


def branch_bias():
    dis, ch = "P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2"
    add("Abstract, 3.4", "charge vs discharge: LAM_Si at the last RPT (%)", f"{rel(dis, ch, 'LAM_Si_pct', [500]).iloc[0]:.1f}",
        "-79")
    add("3.4", "  ... LAM_NE at the last RPT (%)", f"{rel(dis, ch, 'LAM_NE_pct', [500]).iloc[0]:.3f}", "-19")
    a = pd.read_csv(os.path.join(RES, dis, "uncertainty.csv")).set_index("RPT")
    b = pd.read_csv(os.path.join(RES, ch, "uncertainty.csv")).set_index("RPT")
    flips = [r for r in P45B_AGEING if a.LAM_Si_pct_mean[r] > 2 * a.LAM_Si_pct_combined[r]
             and b.LAM_Si_pct_mean[r] < -2 * b.LAM_Si_pct_combined[r]]
    add("Abstract, 3.4", "RPTs where LAM_Si is resolved positive on discharge and resolved negative on charge",
        str(flips), "200 and 300")
    for ref, alt, label, paper in [("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2_discharge-split",
                                    "discharge transition point on both", "-43"),
                                   ("P45B_discharge_2.94-4.2_charge-split", "P45B_charge_2.94-4.2",
                                    "charge transition point on both", "-78"),
                                   ("P45B_discharge_2.94-4.2", "P45B_discharge_2.94-4.2_charge-split",
                                    "discharge branch, own vs charge transition point", "4 (in size)")]:
        add("3.4", f"{label}: LAM_Si at the last RPT (%)", f"{rel(ref, alt, 'LAM_Si_pct', [500]).iloc[0]:.1f}", paper)
    r = mean("P45B_charge_2.94-4.2_discharge-split").LAM_Si_pct[500] / mean(dis).LAM_Si_pct[500]
    add("4.1", "charge LAM_Si with the discharge transition point, as share of discharge (%)", f"{100 * r:.0f}", "57")


# ------------------------------------------------------------------ data
def data_numbers():
    q0 = [m50t_curve(c, 0).Q.max() for c in ("D", "E")]
    add("Table 1", "LG M50T capacity at C/10, beginning of life, two-cell mean (Ah)", f"{np.mean(q0):.2f}", "4.86")

    starts = [p45b_curve(c, r, "charge").V.min() for c in ("cell1", "cell2") for r in P45B_RPTS]
    add("2.2, 4.4", "P45B: highest voltage at which a C/20 charge starts (V)", f"{max(starts):.3f}", "2.93")

    # P45B: C/20 current and the IR part of the charge-discharge gap, 2 |I| R_0.1s.
    for rpt, paper in ((0, "3.7"), (500, "4.0")):
        cur = np.mean([abs(p45b_cc_step(c, rpt, s)["Current / A"].median())
                       for c in ("cell1", "cell2") for s in (P45B_CHARGE_STEP, P45B_DISCHARGE_STEP)])
        r = np.mean([pulse_resistance(c, rpt)[0].mean() for c in ("cell1", "cell2")])
        add("3.1", f"P45B: IR part of the C/20 charge-discharge gap, RPT {rpt} (mV)", f"{2 * cur * r:.1f}", paper)
    diff = max(np.max(np.abs(pulse_resistance(c, r)[0] - pulse_resistance(c, r)[1]))
               for c in ("cell1", "cell2") for r in P45B_RPTS)
    add("3.1", "P45B: largest charge-discharge difference of R_0.1s at one SOC (mOhm)", f"{diff:.2f}", "within 1")

    # LG M50T at beginning of life: slow part of the resistance (10 s vs 0.1 s), and the
    # offset the 10 s resistance would give at the C/10 current at the lowest-SOC pulse.
    ratio, lift = [], []
    for c in ("D", "E"):
        soc, rr = m50t_resistance_ratio(c, 0, 10.0)
        _, r01 = m50t_resistance(c, 0)
        ratio.append(rr.max())
        lift.append(1000 * M50T_POCV_CURRENT_A * (rr * r01)[np.argmin(soc)])
    add("3.1", "LG M50T, BoL: largest 10 s / 0.1 s resistance ratio", f"{max(ratio):.1f}", "up to 2.6")
    add("3.1", "LG M50T, BoL: I * R_10s at the lowest-SOC pulse (mV)", span(lift, "{:.0f}"), "around 40")

    # LG M50T, last RPT, IR-corrected: median |dV/dz_FC| (V per unit of full-cell SOC) where the
    # positive electrode (z > 0.85) and silicon (z < 0.15) are identified.
    hi, lo = [], []
    for c in ("D", "E"):
        z, u, q = fit.prepare_window(m50t_curve(c, 14, True), (2.5, 4.2))
        d = np.abs(np.gradient(fit._savgol(u, 11, 2), z))
        hi.append(np.median(d[z > 0.85]))
        lo.append(np.median(d[z < 0.15]))
    add("3.2", "LG M50T, last RPT: median |dV/dz_FC| at z_FC > 0.85 (V)", span(hi, "{:.2f}"), "0.45-0.46")
    add("3.2", "  ... at z_FC < 0.15 (V)", span(lo, "{:.2f}"), "4.05")


def pulse_resistance(cell, rpt):
    """R_0.1s (mOhm) of the 1C, 30 s discharge and charge pulses of one P45B cell, in test order."""
    d = pd.read_csv(p45b_file(cell, rpt, "rate-and-pulse"))
    t, v, i = d["Test Time / s"].values, d["Voltage / V"].values, d["Current / A"].values
    out = []
    for sign in (-1, 1):
        on = sign * i > 3
        starts = np.where(on[1:] & ~on[:-1])[0] + 1
        r = []
        for k in starts:
            e = k
            while e < len(i) and on[e]:
                e += 1
            if 25 <= t[e - 1] - t[k] <= 35:
                r.append(r_inst(v, i, k))
        out.append(np.array(r))
    return out


def hysteresis_numbers():
    import fig4_hysteresis as f4
    r = f4.analyse()
    G, F = r["G"], r["F"]
    upper = G > G[0] + 0.5 * (G[-1] - G[0])
    add("3.1", "P45B gap, first RPT: largest value (mV)", f"{np.nanmax(F):.0f}", "exceeds 300")
    add("3.1", "  ... range over the upper half of the window (mV)", span(F[upper], "{:.0f}"), "20 to 50")
    add("Fig. 4", "largest change of the fitted shift with a 0.10 or 0.30 Ah window instead of 0.20 Ah (Ah)",
        f"{max(abs(s - r['shift']) for s in r['shift_windows']):.3f}", "at most 0.005")


if __name__ == "__main__":
    resistance_bias()
    window_bias()
    branch_bias()
    data_numbers()
    hysteresis_numbers()
    os.makedirs(paths.OUTPUT, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(os.path.join(paths.OUTPUT, "text_numbers.csv"), index=False)
    print("written to", os.path.join(paths.OUTPUT, "text_numbers.csv"))
