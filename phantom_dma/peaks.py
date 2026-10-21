"""Peak positions of measured and fitted curves, read as incremental capacity dz/dV.

Peaks are found in the measured curve and each is matched to the nearest peak of
the fitted curve. The position error (fitted minus measured, in mV) is used in
Table S1 and Fig. S1.
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def ic_curve(z, v):
    """Smoothed voltage and incremental capacity dz/dV on the fitted grid."""
    vs = savgol_filter(v, 11, 2)
    dvdz = np.gradient(vs, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        return vs, 1.0 / dvdz


def find_ic_peaks(ic, prominence):
    ok = np.isfinite(ic)
    idx = np.where(ok)[0]
    p, _ = find_peaks(np.abs(ic[ok]), prominence=prominence)
    return idx[p]


def match_peaks(z, u, v_model):
    """Measured peaks and their matched fitted peaks.

    Returns (v_meas, ic_meas, v_fit, ic_fit, pairs), where pairs lists
    (index of measured peak, index of matched fitted peak).
    """
    vu, icu = ic_curve(z, u)
    vv, icm = ic_curve(z, v_model)
    prominence = 0.15 * np.nanmax(np.abs(icu[np.isfinite(icu)]))
    pu, pm = find_ic_peaks(icu, prominence), find_ic_peaks(icm, 0.5 * prominence)
    pairs = [(i, pm[np.argmin(np.abs(vv[pm] - vu[i]))] if len(pm) else None) for i in pu]
    return vu, icu, vv, icm, pairs


def peak_errors(results_folder, analysis):
    """Position error of every measured peak, for every cell and RPT of one analysis."""
    rows = []
    fits = os.path.join(results_folder, analysis, "fits")
    for f in sorted(os.listdir(fits)):
        cell, _, rpt = f[:-4].split("_")
        a = np.load(os.path.join(fits, f))
        vu, _, vv, _, pairs = match_peaks(a["z"], a["U_exp"], a["V_model"])
        for i, j in pairs:
            rows.append(dict(Cell=cell, RPT=int(rpt), V_meas=round(vu[i], 3),
                             dV_mV=np.nan if j is None else round(1000 * (vv[j] - vu[i]), 1)))
    return pd.DataFrame(rows)
