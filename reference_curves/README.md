# Half-cell reference curves

Each file has two columns: `z`, the state of lithiation of the electrode material (0 to 1), and
`OCV`, its potential against Li/Li+ in V. Lithiation and delithiation branches are kept apart,
because the fits use the branch that matches the direction of the full-cell curve
(see `phantom_dma/analyses.py`).

| File | Material | Source |
|---|---|---|
| `P45B_PE_NCA_lithiation.csv`, `P45B_PE_NCA_delithiation.csv` | NCA, positive electrode of the Molicel P45B | This work: coin half-cells from a fresh P45B, C/20, 23 °C; also in the Zenodo record doi:10.5281/zenodo.23538218 (Folkson et al., J. Electrochem. Soc. 173 (2026) 010501) |
| `M50T_PE_NMC811_lithiation.csv` | NMC811, positive electrode of the LG M50T | Chen et al., J. Electrochem. Soc. 167 (2020) 080534, doi:10.1149/1945-7111/ab9050 |
| `graphite_lithiation.csv`, `graphite_delithiation.csv` | graphite | Mercer et al., J. Mater. Chem. A 9 (2021) 492-504, doi:10.1039/D0TA10403E |
| `SiO_lithiation.csv`, `SiO_delithiation.csv` | SiO | Kitada et al., J. Am. Chem. Soc. 141 (2019) 7014-7027, doi:10.1021/jacs.9b01589 |

If you use these curves, cite their sources.
