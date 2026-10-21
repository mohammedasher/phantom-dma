"""Full-cell pOCV model for voltage-curve degradation mode analysis (DMA).

The full-cell voltage is the difference between the positive-electrode (PE)
and composite negative-electrode (NE) potentials, each mapped onto the
full-cell state of charge z_FC with a stretch (nu) and a shift (sigma):

    z_NE = nu_NE * z_FC + sigma_NE
    z_PE = 1 - (nu_PE * z_FC + sigma_PE)

The NE is a silicon-graphite blend. Its state of charge at a common potential
U is phi_Si * z_Si(U) + (1 - phi_Si) * z_Gr(U), which is then inverted to give
U_NE(z_NE).

Fitted parameters: theta = (nu_NE, nu_PE, sigma_NE, sigma_PE, phi_Si).
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

import paths

HALFCELL_DIR = paths.REFERENCE_CURVES


def _monotone_interp(x, y):
    """PCHIP interpolator on (x, y) after sorting and removing repeated x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_unique, idx = np.unique(x, return_index=True)
    return PchipInterpolator(x_unique, y[idx])


def read_halfcell(name):
    """Read a half-cell curve with columns z (state of lithiation, 0-1) and OCV (V vs Li/Li+)."""
    return pd.read_csv(os.path.join(HALFCELL_DIR, name))


class CellModel:
    """Full-cell voltage model built from three half-cell curves.

    pe, gr, si : half-cell file names in reference_curves/
    pe_flip    : True if the PE curve is tabulated against lithiation, so the
                 model uses 1 - z (the convention of all curves supplied here)
    offsets_V  : optional (dPE, dGr, dSi) vertical offsets, used by the
                 half-cell uncertainty Monte Carlo
    """

    def __init__(self, pe, gr, si, pe_flip=True, offsets_V=(0.0, 0.0, 0.0)):
        d_pe, d_gr, d_si = offsets_V
        pe_df, gr_df, si_df = read_halfcell(pe), read_halfcell(gr), read_halfcell(si)
        pe_z = (1.0 - pe_df["z"]) if pe_flip else pe_df["z"]
        self.U_pe = _monotone_interp(pe_z, pe_df["OCV"] + d_pe)
        z_gr = _monotone_interp(gr_df["OCV"] + d_gr, gr_df["z"])
        z_si = _monotone_interp(si_df["OCV"] + d_si, si_df["z"])

        # Common potential grid on which the two NE components are blended.
        v_lo = max(z_gr.x[0], z_si.x[0])
        v_hi = min(z_gr.x[-1], z_si.x[-1])
        self.V_common = np.linspace(v_lo, v_hi, 1000)
        self.z_gr_smooth = gaussian_filter1d(z_gr(self.V_common), sigma=2)
        self.z_si_smooth = gaussian_filter1d(z_si(self.V_common), sigma=3)

    def ne_curve(self, phi_si):
        """Composite NE potential as a function of its state of lithiation."""
        z_ne = phi_si * self.z_si_smooth + (1 - phi_si) * self.z_gr_smooth
        valid = ~np.isnan(z_ne)
        z_valid, v_valid = z_ne[valid], self.V_common[valid]
        order = np.argsort(z_valid)
        z_sorted, v_sorted = z_valid[order], v_valid[order]
        z_unique, idx = np.unique(z_sorted, return_index=True)
        v_unique = v_sorted[idx]
        # Trim five points at each end, where the blend is least reliable.
        return PchipInterpolator(z_unique[5:-5], v_unique[5:-5], extrapolate=True)

    def voltage(self, z_fc, theta):
        """Return (U_PE, U_NE, V_cell) at full-cell SOC z_fc."""
        nu_ne, nu_pe, sigma_ne, sigma_pe, phi_si = theta
        z_ne = nu_ne * z_fc + sigma_ne
        z_pe = 1 - (nu_pe * z_fc + sigma_pe)
        u_pe = self.U_pe(z_pe)
        u_ne = self.ne_curve(phi_si)(z_ne)
        return u_pe, u_ne, u_pe - u_ne


def reservoirs(theta, q_fc):
    """Fitted electrode capacities and lithium inventory (Ah) for full-cell capacity q_fc."""
    nu_ne, nu_pe, sigma_ne, sigma_pe, phi_si = theta
    q_ne = q_fc / nu_ne
    q_pe = q_fc / nu_pe
    q_li = q_fc * (sigma_ne / nu_ne + (1 - sigma_pe) / nu_pe)
    return dict(Q_NE=q_ne, Q_PE=q_pe, Q_LI=q_li, Q_Si=phi_si * q_ne, Q_Gr=(1 - phi_si) * q_ne)


def losses(res, res0):
    """Degradation modes (%) relative to the beginning-of-life fit res0."""
    out = {}
    for name, key in [("LAM_PE", "Q_PE"), ("LAM_NE", "Q_NE"), ("LLI", "Q_LI"),
                      ("LAM_Si", "Q_Si"), ("LAM_Gr", "Q_Gr")]:
        out[name + "_pct"] = 100 * (1 - res[key] / res0[key])
    return out
