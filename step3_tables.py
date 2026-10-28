"""Step 3: make Tables S1-S8 of the SI from the results folder.

    python step3_tables.py

Writes to output/tables/: the body rows of each table as LaTeX (TableS1_rows.tex, ...)
and the numbers behind Table S6 as CSV.
Each table shows the mean of the two cells +/- half their range, unless stated.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import paths
from phantom_dma.analyses import HALFCELLS, SCREEN_BOUNDS
from phantom_dma.fit import DEFAULT_BOUNDS
from phantom_dma.model import CellModel, losses
from phantom_dma.peaks import peak_errors
from phantom_dma.uncertainty import paired_difference

RES = paths.RESULTS
OUT = os.path.join(paths.OUTPUT, "tables")
PM = r"\,$\pm$\,"
PARAMS = ["nu_NE", "nu_PE", "sigma_NE", "sigma_PE", "phi_Si"]
SYMBOL = {"nu_NE": r"$\nu_{\mathrm{NE}}$", "nu_PE": r"$\nu_{\mathrm{PE}}$", "sigma_NE": r"$\sigma_{\mathrm{NE}}$",
          "sigma_PE": r"$\sigma_{\mathrm{PE}}$", "phi_Si": r"$\phi_{\mathrm{Si}}$"}
P45B_MAIN = [("P45B_charge_2.94-4.2", "charge 2.94--4.2 V"), ("P45B_discharge_2.5-4.2", "discharge 2.5--4.2 V"),
             ("P45B_discharge_2.94-4.2", "discharge 2.94--4.2 V"), ("P45B_discharge_3.0-4.2", "discharge 3.0--4.2 V")]
M50T = [("M50T_discharge_IR-corrected", "IR-corrected"), ("M50T_discharge_uncorrected", "uncorrected")]
M50T_RPTS = (0, 2, 4, 6, 8, 10, 12, 14)   # tests with pulse data, so both M50T analyses exist


# ------------------------------------------------------------------ helpers
def summary(analysis):
    return pd.read_csv(os.path.join(RES, analysis, "summary.csv"))


def pm(values, fmt="{:.1f}"):
    """'mean +/- half range' of the two cells."""
    v = np.asarray(values, float)
    return fmt.format(v.mean()) + PM + fmt.format((v.max() - v.min()) / 2)


def rmse_low_high(analysis, cell, rpt):
    """Voltage residual (mV) in the lowest and highest third of full-cell SOC."""
    a = np.load(os.path.join(RES, analysis, "fits", f"{cell}_RPT_{rpt:03d}.npz"))
    z, u, vm = a["z"], savgol_filter(a["U_exp"], 11, 2), a["V_model"]
    keep = (z >= 0.02) & (z <= 0.98)
    z, u, vm = z[keep], u[keep], vm[keep]
    low, high = z <= np.quantile(z, 1 / 3), z >= np.quantile(z, 2 / 3)
    rmse = lambda k: 1000 * np.sqrt(np.mean((u[k] - vm[k]) ** 2))
    return rmse(low), rmse(high)


def save(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}_rows.tex"), "w") as f:
        f.write(body + "\n")
    print(f"--- {name}\n{body}\n")


# ------------------------------------------------------------------ SI tables
def table_s1():
    """Fit quality, P45B: RMSE, low- and high-SOC residual, largest peak error (mV)."""
    rows = []
    for analysis, label in P45B_MAIN:
        s = summary(analysis)
        peaks = peak_errors(RES, analysis)
        rows.append(r"\multicolumn{5}{l}{\textit{" + label + r"}} \\")
        for rpt, g in s.groupby("RPT"):
            low, high = zip(*[rmse_low_high(analysis, c, rpt) for c in g.Cell])
            worst = peaks[peaks.RPT == rpt].dV_mV.abs().max()
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(low)} & {pm(high)} & {worst:.0f} \\\\")
    save("TableS1", "\n".join(rows))


def table_s2(rpts=(0, 50, 500)):
    """Fitted parameters, P45B, at beginning of life, the first ageing RPT and the last RPT."""
    rows = []
    for analysis, label in P45B_MAIN:
        s = summary(analysis)
        rows.append(r"\multicolumn{6}{l}{\textit{" + label + r"}} \\")
        for rpt in rpts:
            g = s[s.RPT == rpt]
            rows.append(f"{rpt} & " + " & ".join(pm(g[k], "{:.3f}") for k in PARAMS) + r" \\")
    save("TableS2", "\n".join(rows))


def table_s3():
    """Window screen at the first and last RPT: RMSE, phi_Si, nu_PE; LAM_NE and LAM*_Si at the last RPT."""
    screens = [("P45B_discharge_2.5-4.2", "discharge 2.5--4.2 V", DEFAULT_BOUNDS)]
    screens += [(f"P45B_discharge_{lo}-4.2_screen", f"discharge {lo}--4.2 V", SCREEN_BOUNDS)
                for lo in ("3.2", "3.3", "3.4", "3.5")]
    rows = []
    for analysis, label, bounds in screens:
        s = summary(analysis)
        rows.append(r"\multicolumn{6}{l}{\textit{" + label + r"}} \\")
        for rpt in (0, 500):
            g = s[s.RPT == rpt]
            loss = " & ".join(pm(g[k]) if rpt else "--" for k in ("LAM_NE_pct", "LAM_Si_pct"))
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(g.phi_Si, '{:.3f}')} & {pm(g.nu_PE, '{:.3f}')} & {loss} \\\\")
    save("TableS3", "\n".join(rows))


def table_s4():
    """Fit quality, LG M50T, with and without the IR correction."""
    rows = []
    for analysis, label in M50T:
        s = summary(analysis)
        rows.append(r"\multicolumn{4}{l}{\textit{" + label + r"}} \\")
        for rpt in M50T_RPTS:
            g = s[s.RPT == rpt]
            low, high = zip(*[rmse_low_high(analysis, c, rpt) for c in g.Cell])
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(low)} & {pm(high)} \\\\")
    save("TableS4", "\n".join(rows))


def table_s5():
    """Fitted parameters, LG M50T, first and last test with pulse data."""
    rows = []
    for analysis, label in M50T:
        s = summary(analysis)
        rows.append(r"\multicolumn{6}{l}{\textit{" + label + r"}} \\")
        for rpt in (M50T_RPTS[0], M50T_RPTS[-1]):
            g = s[s.RPT == rpt]
            rows.append(f"{rpt} & " + " & ".join(pm(g[k], "{:.3f}") for k in PARAMS) + r" \\")
    save("TableS5", "\n".join(rows))


def sensitivity_share(model, z, theta, eps=1e-4):
    """Share (%) of sum (dV/dtheta)^2 for phi_Si in the lowest third and nu_PE in the highest third of z."""
    out = {}
    low = z <= z.min() + (z.max() - z.min()) / 3
    high = z >= z.max() - (z.max() - z.min()) / 3
    for k, sel in ((4, low), (1, high)):
        up, down = np.array(theta, float), np.array(theta, float)
        up[k] += eps
        down[k] -= eps
        d = (model.voltage(z, up)[2] - model.voltage(z, down)[2]) / (2 * eps)
        out[k] = 100 * np.sum(d[sel] ** 2) / np.sum(d ** 2)
    return out[4], out[1]


def table_s6():
    """Where the fit learns phi_Si and nu_PE: share of the voltage sensitivity by region (range over cells and RPTs 0, 500)."""
    analyses = [("P45B_discharge_2.5-4.2", "discharge", "discharge 2.5--4.2~V"),
                ("P45B_discharge_2.94-4.2", "discharge", "discharge 2.94--4.2~V"),
                ("P45B_charge_2.94-4.2", "charge", "charge 2.94--4.2~V")]
    rows, raw = [], []
    for analysis, branch, label in analyses:
        model = CellModel(*HALFCELLS[("P45B", branch)])
        phi, nu = [], []
        for cell in ("cell1", "cell2"):
            for rpt in (0, 500):
                a = np.load(os.path.join(RES, analysis, "fits", f"{cell}_RPT_{rpt:03d}.npz"))
                p, n = sensitivity_share(model, a["z"], a["theta"])
                phi.append(p)
                nu.append(n)
                raw.append(dict(Analysis=analysis, Cell=cell, RPT=rpt, phi_Si_lowest_third_pct=p,
                                nu_PE_highest_third_pct=n))
        span = lambda v: f"{min(v):.0f}" if round(min(v)) == round(max(v)) else f"{min(v):.0f}--{max(v):.0f}"
        rows.append(f"{label} & {span(phi)}\\,\\% & {span(nu)}\\,\\% \\\\")
    save("TableS6", "\n".join(rows))
    pd.DataFrame(raw).to_csv(os.path.join(OUT, "TableS6_sensitivity_share.csv"), index=False)


def table_s7():
    """Branch comparison at the last RPT with each branch's own transition point and with a common one."""
    rows_def = [("P45B_discharge_2.94-4.2", "discharge", "own"),
                ("P45B_discharge_2.94-4.2_charge-split", "discharge", "charge"),
                ("P45B_charge_2.94-4.2", "charge", "own"),
                ("P45B_charge_2.94-4.2_discharge-split", "charge", "discharge")]
    rows = []
    for analysis, branch, source in rows_def:
        deltas = [float(np.load(os.path.join(RES, analysis, "fits", f))["delta"])
                  for f in os.listdir(os.path.join(RES, analysis, "fits")) if f.endswith(".npz")]
        u = pd.read_csv(os.path.join(RES, analysis, "uncertainty.csv")).set_index("RPT").loc[500]
        vals = " & ".join(f"{u[k + '_mean']:.1f}" + PM + f"{u[k + '_combined']:.1f}"
                          for k in ("LAM_NE_pct", "LAM_Si_pct", "LAM_Gr_pct", "LLI_pct"))
        rows.append(f"{branch} & {min(deltas):.3f}--{max(deltas):.3f} ({source}) & {vals} \\\\")
    pairs = [("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2", "own"),
             ("P45B_discharge_2.94-4.2_charge-split", "P45B_charge_2.94-4.2", "charge delta on both"),
             ("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2_discharge-split", "discharge delta on both")]
    notes = []
    for ref, alt, label in pairs:
        d = paired_difference(RES, ref, alt, "LAM_Si_pct")
        resolved = [r for r in d.index if r > 0 and abs(d.loc[r, "delta"]) > 2 * d.loc[r, "combined"]]
        notes.append(f"% {label}: dLAM_Si(RPT 500) = {d.loc[500, 'delta']:.1f} +/- {d.loc[500, 'combined']:.1f}; resolved at RPT {resolved}")
    save("TableS7", "\n".join(rows + notes))


def table_s8():
    """M50T losses per RPT with no correction, the 0.1 s correction and the 10 s correction,
    and the RPT-averaged overstatement of LAM_PE without correction (quoted in Section 3.2)."""
    runs = ["M50T_discharge_uncorrected", "M50T_discharge_IR-corrected", "M50T_discharge_IR10s-corrected"]
    s = {r: summary(r) for r in runs}
    rows = []
    for rpt in M50T_RPTS:
        cells = []
        for k in ("LAM_PE_pct", "LAM_NE_pct", "LAM_Si_pct"):
            for r in runs:
                cells.append(pm(s[r][s[r].RPT == rpt][k]))
        rows.append(f"{rpt} & " + " & ".join(cells) + r" \\")
    ageing = list(M50T_RPTS[1:])
    ref, alt = "M50T_discharge_IR-corrected", "M50T_discharge_uncorrected"
    # Two-cell mean over ageing RPTs, with the paired uncertainty averaged in the same way.
    m = s[ref].merge(s[alt], on=["Cell", "RPT"], suffixes=("_r", "_a"))
    m = m[m.RPT.isin(ageing)]
    m["d"] = m["LAM_PE_pct_a"] - m["LAM_PE_pct_r"]
    per_cell = m.groupby("Cell")["d"].mean()
    half_range = (per_cell.max() - per_cell.min()) / 2
    sds = {}
    for cell in sorted(per_cell.index):
        a = pd.read_csv(os.path.join(RES, alt, f"{cell}_mc.csv"))
        b = pd.read_csv(os.path.join(RES, ref, f"{cell}_mc.csv"))
        j = a.merge(b, on=["draw", "RPT"], suffixes=("_a", "_r"))
        j = j[j.RPT.isin(ageing)]
        sds[cell] = (j["LAM_PE_pct_a"] - j["LAM_PE_pct_r"]).groupby(j.draw).mean().std()
    mc = float(np.mean(list(sds.values())))
    rows.append(f"% LAM_PE overstatement averaged over ageing RPTs: two-cell {per_cell.mean():.2f} +/- "
                f"{np.hypot(half_range, mc):.2f}; " + "; ".join(f"{c} {per_cell[c]:.2f} +/- {sds[c]:.2f}" for c in sorted(per_cell.index)))
    save("TableS8", "\n".join(rows))


if __name__ == "__main__":
    for make in (table_s1, table_s2, table_s3, table_s4, table_s5, table_s6, table_s7, table_s8):
        make()
    print("Tables written to", OUT)
