"""Every analysis reported in the paper.

All analyses use the same fitting code. Only the settings below differ: which
cell, which branch of the pOCV (charge or discharge), the voltage window, and
for a few analyses the search limits or the number of optimiser trials.
"""

# Half-cell reference curves for each cell and branch (files in reference_curves/).
# Order: positive electrode, graphite, silicon.
# Discharge fits use the delithiation curves of the negative electrode and the
# lithiation curve of the positive electrode; charge fits use the reverse.
HALFCELLS = {
    ("P45B", "discharge"): ("P45B_PE_NCA_lithiation.csv", "graphite_delithiation.csv", "SiO_delithiation.csv"),
    ("P45B", "charge"): ("P45B_PE_NCA_delithiation.csv", "graphite_lithiation.csv", "SiO_lithiation.csv"),
    ("M50T", "discharge"): ("M50T_PE_NMC811_lithiation.csv", "graphite_delithiation.csv", "SiO_delithiation.csv"),
}

# The two cells of each type. P45B: our names. M50T: cell1 and cell2 are the
# cells called "D" and "E" in the dataset of Kirkaldy et al.
CELLS = {"P45B": {"cell1": "cell1", "cell2": "cell2"},
         "M50T": {"cell1": "D", "cell2": "E"}}

# Lower voltage limit of the range that both P45B branches cover: the highest
# voltage at which any C/20 charge starts (2.933 V), rounded up.
P45B_COMMON_LOWER_V = 2.94

# Wider search limits used for the window screens (SI of the paper):
# nu_NE, nu_PE, sigma_NE, sigma_PE, phi_Si.
SCREEN_BOUNDS = [(0.4, 1.05), (0.4, 1.05), (-0.6, 0.5), (-0.2, 0.6), (0.0, 0.3)]

ANALYSES = {
    # Main analyses (Figs. 5-7, Table 2).
    "P45B_discharge_2.5-4.2": dict(cell="P45B", branch="discharge", window=(2.5, 4.2)),
    "P45B_discharge_3.0-4.2": dict(cell="P45B", branch="discharge", window=(3.0, 4.2)),
    "P45B_discharge_2.94-4.2": dict(cell="P45B", branch="discharge", window=(P45B_COMMON_LOWER_V, 4.2)),
    "P45B_charge_2.94-4.2": dict(cell="P45B", branch="charge", window=(P45B_COMMON_LOWER_V, 4.2)),
    "M50T_discharge_IR-corrected": dict(cell="M50T", branch="discharge", window=(2.5, 4.2), ir_corrected=True),
    "M50T_discharge_uncorrected": dict(cell="M50T", branch="discharge", window=(2.5, 4.2), ir_corrected=False,
                                       mc_rpts=(0, 2, 4, 6, 8, 10, 12, 14)),
    # Window screens (Table S3): three optimiser trials per RPT and wider limits.
    "P45B_discharge_2.5-4.0_screen": dict(cell="P45B", branch="discharge", window=(2.5, 4.0), n_trials=3,
                                          bounds=SCREEN_BOUNDS),
    "P45B_discharge_3.2-4.2_screen": dict(cell="P45B", branch="discharge", window=(3.2, 4.2), n_trials=3,
                                          bounds=SCREEN_BOUNDS),
    "P45B_discharge_3.5-4.2_screen": dict(cell="P45B", branch="discharge", window=(3.5, 4.2), n_trials=3,
                                          bounds=SCREEN_BOUNDS),
    # Check in the SI: the charge branch fitted with dV/dz over the whole curve.
    "P45B_charge_2.94-4.2_dvdz_whole_curve": dict(cell="P45B", branch="charge",
                                                  window=(P45B_COMMON_LOWER_V, 4.2), dvdz_whole_curve=True),
}

# Analyses with the half-cell Monte Carlo.
WITH_MONTE_CARLO = ["P45B_discharge_2.5-4.2", "P45B_discharge_3.0-4.2", "P45B_discharge_2.94-4.2",
                    "P45B_charge_2.94-4.2", "M50T_discharge_IR-corrected", "M50T_discharge_uncorrected"]
