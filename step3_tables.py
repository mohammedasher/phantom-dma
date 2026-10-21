"""Step 3: make Table 2 of the paper and Tables S1-S7 of the SI from the results folder.

    python step3_tables.py

Writes to output/tables/: the body rows of each table as LaTeX (Table2_rows.tex,
TableS1_rows.tex, ...) and the numbers behind Table 2 and Table S7 as CSV.
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


def constrained(group):
    return "yes" if group.constraint_active.fillna("").astype(str).str.len().gt(0).any() else ""


def save(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}_rows.tex"), "w") as f:
        f.write(body + "\n")
    print(f"--- {name}\n{body}\n")


# ------------------------------------------------------------------ Table 2
MODES = [("LAM_PE_pct", r"$\mathrm{LAM}_{\mathrm{PE}}$"), ("LLI_pct", "LLI"), ("LAM_NE_pct", r"$\mathrm{LAM}_{\mathrm{NE}}$"),
         ("LAM_Gr_pct", r"$\mathrm{LAM}^{*}_{\mathrm{Gr}}$"), ("LAM_Si_pct", r"$\mathrm{LAM}^{*}_{\mathrm{Si}}$")]
DECISIONS = [("M50T_discharge_IR-corrected", "M50T_discharge_uncorrected", 14, "IR correction omitted (LG M50T)"),
             ("P45B_discharge_2.5-4.2", "P45B_discharge_3.0-4.2", 500, "Lower cut-off raised to 3.0 V (P45B)"),
             ("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2", 500, "Charge branch instead of discharge (P45B)")]


def table2():
    """Change in each mode at the last RPT: alternative minus reference analysis (percentage points).

    The uncertainty is paired (see phantom_dma/uncertainty.py). Changes smaller than
    twice their uncertainty are set in italics. Relative change = change / reference
    value; for the LG M50T, the mean of the relative change over the ageing RPTs.
    """
    rows, raw = [], []
    for key, label in MODES:
        cells = []
        for ref, alt, rpt, decision in DECISIONS:
            a, b = summary(ref), summary(alt)
            r = a[a.RPT == rpt][key].mean()
            d = b[b.RPT == rpt][key].mean() - r
            p = paired_difference(RES, ref, alt, key).loc[rpt]
            if ref.startswith("M50T"):
                rel = np.mean([100 * (b[b.RPT == k][key].mean() - a[a.RPT == k][key].mean()) / a[a.RPT == k][key].mean()
                               for k in sorted(set(a.RPT) & set(b.RPT)) if 0 < k <= rpt])
                rel_text = f"{rel:+.1f}" if abs(rel) < 10 else f"{rel:+.0f}"
            else:
                rel = 100 * d / r
                rel_text = "--" if abs(r) < 0.5 else (f"{rel:+.1f}" if abs(rel) < 10 else f"{rel:+.0f}")
            text = f"{d:+.1f}" + PM + f"{p.combined:.1f}"
            if abs(d) < 2 * p.combined:
                text = r"\textit{" + text + "}"
            cells.append(f"{text} & {rel_text}")
            raw.append({"Decision": decision, "RPT": rpt, "Mode": key.replace("_pct", ""), "Delta / pp": d,
                        "Cell Half-Range / pp": p.cell_half_range, "Monte Carlo SD / pp": p.mc_sd,
                        "Paired Uncertainty / pp": p.combined})
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    body = re.sub(r"([&{] ?)([+-])(?=\d)", lambda m: m.group(1) + "$" + m.group(2) + "$", body)
    save("Table2", body)
    pd.DataFrame(raw).to_csv(os.path.join(OUT, "Table2_bias_last_RPT.csv"), index=False)


# ------------------------------------------------------------------ SI tables
def table_s1():
    """Fit quality, P45B: RMSE, low- and high-SOC residual, V_max error, largest peak error (mV)."""
    rows = []
    for analysis, label in P45B_MAIN:
        s = summary(analysis)
        peaks = peak_errors(RES, analysis)
        rows.append(r"\multicolumn{7}{l}{\textit{" + label + r"}} \\")
        for rpt, g in s.groupby("RPT"):
            low, high = zip(*[rmse_low_high(analysis, c, rpt) for c in g.Cell])
            worst = peaks[peaks.RPT == rpt].dV_mV.abs().max()
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(low)} & {pm(high)} & {pm(g.Vmax_err_mV)} & {worst:.0f} & "
                        f"{constrained(g)} \\\\")
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


def table_s3(tol=0.005):
    """Window screen at the first and last RPT; parameters within 0.005 of a search limit are listed."""
    screens = [("P45B_discharge_2.5-4.0_screen", "discharge 2.5--4.0 V", SCREEN_BOUNDS),
               ("P45B_discharge_2.5-4.2", "discharge 2.5--4.2 V", DEFAULT_BOUNDS),
               ("P45B_discharge_3.2-4.2_screen", "discharge 3.2--4.2 V", SCREEN_BOUNDS),
               ("P45B_discharge_3.5-4.2_screen", "discharge 3.5--4.2 V", SCREEN_BOUNDS)]
    rows = []
    for analysis, label, bounds in screens:
        s = summary(analysis)
        rows.append(r"\multicolumn{5}{l}{\textit{" + label + r"}} \\")
        for rpt in (0, 500):
            g = s[s.RPT == rpt]
            hit = [SYMBOL[k] for k, (lo, hi) in zip(PARAMS, bounds)
                   if ((g[k] - lo).abs() < tol).any() or ((g[k] - hi).abs() < tol).any()]
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(g.phi_Si, '{:.3f}')} & {pm(g.nu_PE, '{:.3f}')} & "
                        f"{', '.join(hit)} \\\\")
    save("TableS3", "\n".join(rows))


def table_s4():
    """Fit quality, LG M50T, with and without the IR correction."""
    rows = []
    for analysis, label in M50T:
        s = summary(analysis)
        rows.append(r"\multicolumn{6}{l}{\textit{" + label + r"}} \\")
        for rpt in M50T_RPTS:
            g = s[s.RPT == rpt]
            low, high = zip(*[rmse_low_high(analysis, c, rpt) for c in g.Cell])
            rows.append(f"{rpt} & {pm(g.RMSE_mV)} & {pm(low)} & {pm(high)} & {pm(g.Vmax_err_mV)} & {constrained(g)} \\\\")
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


def table_s6():
    """Silicon-resolved loss re-referenced to the first ageing RPT (RPT 50), P45B 2.94-4.2 V."""
    rows = []
    per_branch = {}
    for analysis in ("P45B_discharge_2.94-4.2", "P45B_charge_2.94-4.2"):
        s = summary(analysis)
        out = []
        for cell, g in s.groupby("Cell"):
            ref = g[g.RPT == 50].iloc[0]
            ref_cap = {k: ref[k + "_Ah"] for k in ("Q_NE", "Q_PE", "Q_LI", "Q_Si", "Q_Gr")}
            for _, row in g.iterrows():
                out.append(dict(Cell=cell, RPT=row.RPT,
                                **losses({k: row[k + "_Ah"] for k in ref_cap}, ref_cap)))
        per_branch[analysis] = pd.DataFrame(out)
    for rpt in (100, 200, 300, 400, 500):
        cells = []
        for analysis, t in per_branch.items():
            v = t[t.RPT == rpt].LAM_Si_pct.values
            m, h = v.mean(), (v.max() - v.min()) / 2
            cells.append((f"${m:.1f}$" if m < 0 else f"{m:.1f}") + PM + f"{h:.1f}")
        rows.append(f"{rpt} & " + " & ".join(cells) + r" \\")
    save("TableS6", "\n".join(rows))


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


def table_s7():
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
    save("TableS7", "\n".join(rows))
    pd.DataFrame(raw).to_csv(os.path.join(OUT, "TableS7_sensitivity_share.csv"), index=False)


if __name__ == "__main__":
    for make in (table2, table_s1, table_s2, table_s3, table_s4, table_s5, table_s6, table_s7):
        make()
    print("Tables written to", OUT)
