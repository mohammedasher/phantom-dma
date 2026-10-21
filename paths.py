"""Where the input data are. Edit the two folder paths below if your data are elsewhere.

P45B_DATA  the Zenodo record 10.5281/zenodo.23538218 (Molicel P45B data), downloaded
           as it is: a folder holding P45B_25degC_cell1_RPT000_pOCV-C20.csv and the
           other files of that record.
M50T_DATA  the folder "Expt 5 - Standard Cycle Aging (Control)" of the Zenodo record
           10.5281/zenodo.10637534 (LG M50T data of Kirkaldy et al.), unzipped as it is.
           If unzipping created one more folder level, it is found automatically.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

P45B_DATA = os.path.join(HERE, "data", "P45B")
M50T_DATA = os.path.join(HERE, "data", "M50T", "Expt 5 - Standard Cycle Aging (Control)")

# Some unzip tools create the folder twice ("Expt 5 ...\Expt 5 ..."); use the inner one then.
_inner = os.path.join(M50T_DATA, "Expt 5 - Standard Cycle Aging (Control)")
if not os.path.isdir(os.path.join(M50T_DATA, "Summary Data")) and os.path.isdir(_inner):
    M50T_DATA = _inner

REFERENCE_CURVES = os.path.join(HERE, "reference_curves")   # half-cell curves (in this repository)
RESULTS = os.path.join(HERE, "results")                     # fitted results (in this repository)
OUTPUT = os.path.join(HERE, "output")                       # tables and figures are written here
