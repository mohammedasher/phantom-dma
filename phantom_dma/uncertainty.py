"""Half-cell voltage-offset uncertainty by Monte Carlo.

Each draw shifts the PE, graphite and silicon curves by independent normal
offsets of standard deviation SIGMA_U_MV, refits every RPT from the nominal
parameters (three short warm-started trials, best kept), and recomputes the
degradation modes against that draw's own beginning-of-life refit.
"""
import os

import numpy as np
import pandas as pd

from .fit import fit_curve, prepare_window
from .model import CellModel, losses, reservoirs

SIGMA_U_MV = 2.0
N_DRAWS = 15
N_TRIALS_MC = 3
SEED = 12345


def run_mc(halfcells, curves, branch, window, nominal, cell_index=0, sigma_mv=SIGMA_U_MV,
           n_draws=N_DRAWS, log=print):
    """nominal: DataFrame from fit_sequence for this cell. Returns one row per draw and RPT."""
    rng = np.random.default_rng(SEED + cell_index)
    prepared = {r: prepare_window(curves[r], window) for r in sorted(curves)}
    nominal = nominal.set_index("RPT")
    rows = []
    for draw in range(n_draws):
        offsets = rng.normal(0.0, sigma_mv, 3) / 1000.0
        model = CellModel(*halfcells, offsets_V=offsets)
        res0, q_ref = None, None
        for rpt, (z, u, q_fc) in prepared.items():
            n = nominal.loc[rpt]
            seed_theta = [n.nu_NE, n.nu_PE, n.sigma_NE, n.sigma_PE, n.phi_Si]
            theta, _, _ = fit_curve(model, z, u, q_fc, branch, seed_theta, q_ref=q_ref, tol=1.0,
                                    n_trials=N_TRIALS_MC, pop_size=40, spread=0.02, maxiter=300,
                                    seed_offset=(777, 888))
            r = reservoirs(theta, q_fc)
            if res0 is None:
                res0, q_ref = r, (r["Q_NE"], r["Q_PE"], r["Q_LI"])
            rows.append(dict(draw=draw, RPT=rpt, **losses(r, res0),
                             dPE_mV=offsets[0] * 1e3, dGr_mV=offsets[1] * 1e3, dSi_mV=offsets[2] * 1e3))
        log(f"  draw {draw + 1}/{n_draws} done")
    return pd.DataFrame(rows)


def combine(summary, mc):
    """Mean of the two cells, with half their range and the MC standard deviation combined in quadrature."""
    modes = ["LAM_PE_pct", "LAM_NE_pct", "LLI_pct", "LAM_Si_pct", "LAM_Gr_pct"]
    mean = summary.groupby("RPT")[modes].mean()
    half_range = (summary.groupby("RPT")[modes].max() - summary.groupby("RPT")[modes].min()) / 2
    mc_sd = mc.groupby(["Cell", "RPT"])[modes].std().groupby("RPT").mean()
    total = np.sqrt(half_range ** 2 + mc_sd ** 2)
    out = mean.add_suffix("_mean").join(half_range.add_suffix("_cell_half_range")).join(
        mc_sd.add_suffix("_mc_sd")).join(total.add_suffix("_combined"))
    return out.reset_index()


def paired_difference(results, ref, alt, key):
    """Difference alt - ref of one mode between two analyses of the same cells, per RPT.

    The analyses share cells, measured curves and Monte Carlo offset draws, so the
    uncertainty of the difference comes from the differences themselves: half the
    range over cells of the within-cell difference, and the standard deviation over
    draws of the within-draw difference (averaged over cells), in quadrature.
    Returns a DataFrame indexed by RPT: delta, cell_half_range, mc_sd, combined.
    """
    a = pd.read_csv(os.path.join(results, ref, "summary.csv"))
    b = pd.read_csv(os.path.join(results, alt, "summary.csv"))
    m = a.merge(b, on=["Cell", "RPT"], suffixes=("_r", "_a"))
    m["d"] = m[key + "_a"] - m[key + "_r"]
    g = m.groupby("RPT")["d"]
    out = pd.DataFrame({"delta": g.mean(), "cell_half_range": (g.max() - g.min()) / 2})
    sds = []
    for cell in sorted(m.Cell.unique()):
        pa, pb = (os.path.join(results, r, f"{cell}_mc.csv") for r in (ref, alt))
        if os.path.exists(pa) and os.path.exists(pb):
            ma, mb = pd.read_csv(pa), pd.read_csv(pb)
            if key not in ma.columns:  # e.g. SOH: measured, no Monte Carlo term
                continue
            j = ma.merge(mb, on=["draw", "RPT"], suffixes=("_r", "_a"))
            sds.append((j[key + "_a"] - j[key + "_r"]).groupby(j.RPT).std())
    out["mc_sd"] = pd.concat(sds, axis=1).mean(axis=1) if sds else np.nan
    out["combined"] = np.sqrt(out.cell_half_range ** 2 + out.mc_sd.fillna(0) ** 2)
    return out
