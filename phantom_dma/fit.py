"""Objective function and optimiser for one pOCV curve, and the ageing sequence.

Objective function (Methods of the paper):
  0.2 * RMSE(V) on one side of the transition point delta
+ 0.8 * Huber RMSE(dV/dz) on the other side
+ a one-sided quadratic penalty once the modelled voltage extremes miss the
  measured ones by more than 30 mV.
For discharge the voltage term is applied below delta; for charge, above it.
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from .model import reservoirs

# Search limits for nu_NE, nu_PE, sigma_NE, sigma_PE, phi_Si (Methods and SI of the paper).
DEFAULT_BOUNDS = [(0.7, 1.05), (0.7, 1.05), (-0.6, 0.2), (-0.2, 0.6), (0.0, 0.3)]
BOUNDS = list(DEFAULT_BOUNDS)


def set_bounds(bounds=None):
    """Use other search limits (the window screens), or the defaults when bounds is None."""
    global BOUNDS
    BOUNDS = list(bounds or DEFAULT_BOUNDS)
INIT_CENTER = [0.8314, 0.9422, 0.0219, 0.0461, 0.1867]
N_TRIALS = 10          # independent trials per RPT; the best (lowest RMSE) is kept
Z_LO, Z_HI = 0.02, 0.98
BIG_COST = 1e6
ENDPOINT_TOL_V = 0.03
# Optional speed-up: set the environment variable DMA_TRIAL_WORKERS to, say, 4 to run
# four trials at once (Linux and macOS). The result is identical: each trial has its own seeds.
TRIAL_WORKERS = int(os.environ.get("DMA_TRIAL_WORKERS", "1"))


# ---------------------------------------------------------------- utilities
def _grad(x):
    g = np.gradient(np.asarray(x, float))
    g[np.abs(g) < 1e-12] = 1e-12
    return g


def _savgol(y, window=11, poly=2, deriv=0):
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return y.copy() if deriv == 0 else np.zeros_like(y)
    w = int(window) | 1
    w = min(w, n if n % 2 == 1 else n - 1)
    min_w = poly + 2
    if min_w % 2 == 0:
        min_w += 1
    w = max(w, min_w)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= poly or w < 3:
        return y.copy() if deriv == 0 else np.gradient(y)
    return savgol_filter(y, window_length=w, polyorder=poly, deriv=deriv)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _huber_rmse(a, b, delta=0.02):
    r = np.asarray(a, float) - np.asarray(b, float)
    ar = np.abs(r)
    return float(np.sqrt(np.mean(np.where(ar <= delta, 0.5 * r * r, delta * (ar - 0.5 * delta)))))


def _average_duplicates(x_src, y_src, x_tgt):
    x_src, y_src = np.asarray(x_src, float), np.asarray(y_src, float)
    order = np.argsort(x_src)
    xs, ys = x_src[order], y_src[order]
    xu, inv = np.unique(xs, return_inverse=True)
    total = np.zeros_like(xu)
    count = np.zeros_like(xu)
    np.add.at(total, inv, ys)
    np.add.at(count, inv, 1.0)
    if len(xu) < 2:
        return np.full_like(np.asarray(x_tgt, float), np.nan)
    return np.interp(x_tgt, xu, total / np.maximum(count, 1.0))


# --------------------------------------------------------- data preparation
def prepare_window(curve, v_window, step=5, n_grid=500):
    """Restrict a pOCV curve to v_window and resample it on a uniform SOC grid.

    curve: DataFrame with columns V, SOC (full-cell SOC, 1 = full) and Q (Ah).
    SOC is renormalised to [0, 1] over the retained slice, so a reduced window
    is fitted on its own full-cell coordinate.
    """
    df = curve.iloc[::step].copy()
    lo, hi = sorted(v_window)
    keep = np.isfinite(df["V"]) & np.isfinite(df["SOC"]) & (df["V"] >= lo) & (df["V"] <= hi)
    w = df.loc[keep].sort_values("SOC")
    s_min, s_max = w["SOC"].min(), w["SOC"].max()
    w = w.assign(SOC=(w["SOC"] - s_min) / (s_max - s_min))
    z = np.linspace(w["SOC"].min(), w["SOC"].max(), n_grid)
    u = interp1d(w["SOC"], w["V"], kind="linear", bounds_error=False, fill_value="extrapolate")(z)
    return z, u, float(w["Q"].max())


def transition_point(z, u, direction):
    """Transition point delta, kept 0.04 away from the ends of the curve.

    On charge, delta is where the measured dV/dz is smallest. On discharge the
    sign of dV/dz is reversed, so delta falls at the steepest point, the knee at
    the low-SOC end.
    """
    dvdz = _grad(_savgol(u, 11, 2)) / _grad(z)
    metric = (-1.0 if direction == "discharge" else 1.0) * dvdz
    mask = (z >= Z_LO) & (z <= Z_HI) & np.isfinite(metric)
    delta = float(z[mask][np.nanargmin(metric[mask])])
    return min(max(delta, Z_LO + 0.02), Z_HI - 0.02)


def term_masks(z, delta, direction):
    """Voltage term on one side of delta, dV/dz term on the other."""
    if direction == "discharge":
        v_mask = (z >= Z_LO) & (z <= delta)
        d_mask = (z >= delta) & (z <= 1.0)
    else:
        v_mask = (z >= delta) & (z <= Z_HI)
        d_mask = (z >= Z_LO) & (z <= delta)
    return v_mask, d_mask


# ------------------------------------------------------------- feasibility
def feasible(theta, z, q_fc=None, q_ref=None, tol=1.002):
    """Mapped electrode coordinates stay in [0, 1]; reservoirs do not exceed BoL.

    q_ref = (Q_NE, Q_PE, Q_LI) at beginning of life. tol allows 0.2 % slack on the
    first pass; RPTs where the bound is active are refitted with tol = 1.0.
    """
    nu_ne, nu_pe, sigma_ne, sigma_pe, _ = theta
    z0, z1 = np.min(z), np.max(z)
    for v in (nu_ne * z0 + sigma_ne, nu_ne * z1 + sigma_ne,
              1 - (nu_pe * z1 + sigma_pe), 1 - (nu_pe * z0 + sigma_pe)):
        if not 0 <= v <= 1:
            return False
    if q_fc is not None and q_ref is not None:
        r = reservoirs(theta, q_fc)
        if (r["Q_PE"] > q_ref[1] * tol or r["Q_NE"] > q_ref[0] * tol or r["Q_LI"] > q_ref[2] * tol):
            return False
    return True


# --------------------------------------------------------------- objective
def make_objective(model, z, u, direction, delta, q_fc=None, q_ref=None, tol=1.002):
    u_s = _savgol(u, 11, 2)
    u_min, u_max = np.nanmin(u), np.nanmax(u)
    v_mask, d_mask = term_masks(z, delta, direction)
    enough = v_mask.sum() >= 6 and d_mask.sum() >= 6
    z_d = z[d_mask]
    dz = _grad(z_d)
    dvdz_meas = _savgol(u_s[d_mask], 11, 2, 1) / dz

    def cost(theta):
        if not enough or not feasible(theta, z, q_fc, q_ref, tol):
            return BIG_COST
        try:
            v_model = model.voltage(z, theta)[2]
        except Exception:
            return BIG_COST
        if v_model is None or np.any(~np.isfinite(v_model)):
            return BIG_COST
        miss_lo = abs(np.nanmin(v_model) - u_min)
        miss_hi = abs(np.nanmax(v_model) - u_max)
        penalty = 5000.0 * (max(0.0, miss_lo - ENDPOINT_TOL_V) ** 2 + max(0.0, miss_hi - ENDPOINT_TOL_V) ** 2)
        rmse_v = _rmse(u_s[v_mask], v_model[v_mask])
        dvdz_model = _average_duplicates(z_d, _savgol(v_model[d_mask], 11, 2, 1) / dz, z_d)
        if np.any(~np.isfinite(dvdz_model)):
            return BIG_COST
        return 0.2 * rmse_v + 0.8 * _huber_rmse(dvdz_meas, dvdz_model) + penalty

    return cost


def fit_quality(model, z, u, theta, direction, delta):
    """Residual diagnostics reported in Tables S1 and S4."""
    v_model = model.voltage(z, theta)[2]
    u_s = _savgol(u, 11, 2)
    full = (z >= Z_LO) & (z <= Z_HI)

    def region(mask):
        return 1000 * _rmse(u_s[mask], v_model[mask]) if mask.sum() > 3 else np.nan

    return dict(RMSE_mV=1000 * _rmse(u_s[full], v_model[full]),
                RMSE_lowSOC_mV=region(z < 0.05), RMSE_midSOC_mV=region((z >= 0.05) & (z <= 0.95)),
                RMSE_hiSOC_mV=region(z > 0.95),
                Vmin_err_mV=1000 * abs(np.nanmin(v_model) - np.nanmin(u)),
                Vmax_err_mV=1000 * abs(np.nanmax(v_model) - np.nanmax(u)),
                V_model=v_model)


def _initial_population(center, size, spread, seed):
    """Gaussian cloud around center (scaled to the bounds); the first member is center itself."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    scale = (hi - lo) * spread
    pop = np.array([np.clip(center + rng.normal(0, scale), lo, hi) for _ in range(size)])
    pop[0] = np.clip(center, lo, hi)
    return pop


# ------------------------------------------------------------------ fitting
def fit_curve(model, z, u, q_fc, direction, center, q_ref=None, tol=1.002,
              n_trials=N_TRIALS, pop_size=80, spread=0.10, maxiter=800, seed_offset=(3, 7),
              dvdz_whole_curve=False):
    """Fit one curve from n_trials seeded differential-evolution runs; keep the lowest RMSE."""
    delta = transition_point(z, u, direction)
    if dvdz_whole_curve:
        # Check in the SI only: put delta at the end that gives the dV/dz term the
        # whole curve, as happens on discharge.
        delta = Z_LO + 0.02 if direction == "discharge" else Z_HI - 0.02
    cost = make_objective(model, z, u, direction, delta, q_fc, q_ref, tol)
    global _TRIAL_CTX
    _TRIAL_CTX = (cost, np.asarray(center, float), pop_size, spread, maxiter, seed_offset, list(BOUNDS))
    if TRIAL_WORKERS > 1 and n_trials > 1:
        with mp.get_context("fork").Pool(min(TRIAL_WORKERS, n_trials)) as pool:
            xs = pool.map(_run_trial, range(n_trials))
    else:
        xs = [_run_trial(k) for k in range(n_trials)]
    best = None
    for x in xs:
        q = fit_quality(model, z, u, x, direction, delta)
        if best is None or q["RMSE_mV"] < best[1]["RMSE_mV"]:
            best = (x, q)
    return best[0], best[1], delta


_TRIAL_CTX = None


def _run_trial(k):
    """One seeded differential-evolution trial; reads its inputs from _TRIAL_CTX (fork-inherited)."""
    cost, center, pop_size, spread, maxiter, seed_offset, bounds = _TRIAL_CTX
    pop = _initial_population(center, pop_size, spread, seed=1000 * k + seed_offset[0])
    res = differential_evolution(cost, bounds=bounds, init=pop, maxiter=maxiter, popsize=15, tol=8e-6,
                                 strategy="rand1bin", polish=True, updating="deferred", workers=1,
                                 disp=False, seed=2000 * k + seed_offset[1])
    return res.x


def fit_sequence(model, curves, direction, v_window, n_trials=N_TRIALS, log=print, dvdz_whole_curve=False):
    """Fit an ageing sequence {rpt: curve}, warm-starting each RPT from the previous one.

    The first RPT is the beginning-of-life reference. If the monotonicity bound
    is active at an RPT, that RPT is refitted with the hard bound (tol = 1.0).
    Returns a DataFrame with parameters, reservoirs and fit quality per RPT.
    """
    rows, center, q_ref, arrays = [], INIT_CENTER, None, {}
    for rpt in sorted(curves):
        z, u, q_fc = prepare_window(curves[rpt], v_window)
        theta, q, delta = fit_curve(model, z, u, q_fc, direction, center, q_ref, 1.002, n_trials,
                                    dvdz_whole_curve=dvdz_whole_curve)
        r = reservoirs(theta, q_fc)
        hit = []
        if q_ref is not None:
            for name, key, ref in [("NE", "Q_NE", q_ref[0]), ("PE", "Q_PE", q_ref[1]), ("LI", "Q_LI", q_ref[2])]:
                if r[key] >= ref * 1.002 * (1 - 1e-4):
                    hit.append(name)
            if hit:
                theta, q, delta = fit_curve(model, z, u, q_fc, direction, center, q_ref, 1.0, n_trials,
                                            dvdz_whole_curve=dvdz_whole_curve)
                r = reservoirs(theta, q_fc)
        center = list(theta)
        if q_ref is None:
            q_ref = (r["Q_NE"], r["Q_PE"], r["Q_LI"])
        arrays[rpt] = dict(z=z, U_exp=u, V_model=q.pop("V_model"), theta=theta, Q_exp_Ah=q_fc, delta=delta)
        rows.append(dict(RPT=rpt, nu_NE=theta[0], nu_PE=theta[1], sigma_NE=theta[2], sigma_PE=theta[3],
                         phi_Si=theta[4], Q_exp_Ah=q_fc, **{k + "_Ah": v for k, v in r.items()}, **q,
                         constraint_active=",".join(hit)))
        log(f"  RPT {rpt}: RMSE {q['RMSE_mV']:.2f} mV, theta {np.round(theta, 4)}")
    return pd.DataFrame(rows), arrays
