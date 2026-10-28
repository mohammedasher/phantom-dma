# phantom-dma

Code for the paper

> M. Asheruddin N, M. Leal De Souza, C. Folkson, G. Offer, M. Marinescu,
> *Phantom LAM and LLI: Resistance and Hysteresis Bias in Voltage-Curve Degradation Mode Analysis*
> (submitted to Journal of Power Sources; preprint arXiv:2512.19773).

It fits degradation modes (loss of lithium inventory, LLI, and loss of active material,
LAM) to low-rate voltage curves of two commercial cells with silicon-graphite negative
electrodes, and makes every table and data figure of the paper.

You can use it at two levels:

- **Quick (minutes):** make the tables and figures from the fitted results that are already
  in this repository (steps 3 and 4).
- **Full (hours to days):** refit everything from the raw data yourself (steps 1 and 2), then
  make the tables and figures.

---

## 1. Install

You need Python 3.10 or newer.

```bash
git clone https://github.com/mohammedasher/phantom-dma.git
cd phantom-dma
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

Two public datasets are used. Put them in a folder `data/` inside this repository, as below.
(If you keep them somewhere else, change the two paths at the top of `paths.py`.)

```
phantom-dma/
  data/
    P45B/                                   <- Molicel P45B record (all its files)
      P45B_25degC_cell1_RPT000_pOCV-C20.csv
      ...
    M50T/
      Expt 5 - Standard Cycle Aging (Control)/   <- LG M50T data of Kirkaldy et al.
        Processed Timeseries Data/
        Summary Data/
```

| Folder | Where to get it |
|---|---|
| `data/P45B/` | Zenodo, [doi:10.5281/zenodo.23538218](https://doi.org/10.5281/zenodo.23538218). Download all files into `data/P45B/`. |
| `data/M50T/` | Zenodo, [doi:10.5281/zenodo.10637534](https://doi.org/10.5281/zenodo.10637534) (Kirkaldy et al., 2024). Download `Expt 5 - Standard Cycle Aging (Control).zip` (about 10 GB) and unzip it into `data/M50T/`. |

Which steps need which data:

| Step | P45B data | M50T data |
|---|---|---|
| 1, 2 (fitting) | yes | yes |
| 3 (tables) | no | no |
| 4 (figures) | Figs. 1, 2, 4 | Figs. 2, 3, 5 |
| 5 (numbers in the text) | yes | yes |

The half-cell reference curves are in `reference_curves/` (see the README there).

## 3. Run

Run these from the `phantom-dma` folder.

### Step 1: fit (optional; results are included)

```bash
python step1_fit.py --all              # every analysis in the paper
python step1_fit.py P45B_discharge_2.5-4.2   # or just one analysis
```

Each analysis is fitted to both cells of its type, one RPT after another. One cell takes
20 to 50 minutes. `--jobs 4` runs four cells at once if your computer has the cores.
The analyses are listed, with their settings, in `phantom_dma/analyses.py`.

### Step 2: uncertainty (optional; results are included)

```bash
python step2_monte_carlo.py --all
```

This repeats each fit with the half-cell curves shifted by random offsets (standard
deviation 2 mV, 50 draws per cell). It takes several hours per analysis on one core; with
`--jobs` and a computing cluster it is much faster.

### Step 3: tables

```bash
python step3_tables.py
```

Writes the rows of Tables S1-S8 to `output/tables/` as LaTeX, and the numbers behind
Table S6 as CSV. Lines starting with `%` in a rows file are notes (for example the averages
quoted in the caption of Table S8), not table rows.

### Step 4: figures

```bash
python step4_figures.py               # all figures
python figures/fig_bias_modes.py 6    # or one figure
```

Figures go to `output/figures/`.

### Step 5: numbers in the text

```bash
python step5_text_numbers.py
```

Computes the numbers quoted in the text of the paper that are not read directly off a
table or figure (for example the relative biases in the abstract), and writes them to
`output/text_numbers.csv`, each with where it appears and the value as written in the paper.
Run step 3 first.

## 4. What makes each table and figure

| In the paper | Script | Uses |
|---|---|---|
| Fig. 1 | `figures/fig1_fit_workflow.py` | `results/P45B_discharge_2.5-4.2`, P45B data |
| Fig. 2 | `figures/fig2_resistance.py` | P45B and M50T data |
| Fig. 3 | `figures/fig3_ir_correction.py` | M50T data |
| Fig. 4 | `figures/fig4_hysteresis.py` | P45B data |
| Fig. 5 | `figures/fig_bias_modes.py 5` | `results/M50T_discharge_IR-corrected`, `results/M50T_discharge_uncorrected`, M50T summary files |
| Fig. 6 | `figures/fig_bias_modes.py 6` | `results/P45B_discharge_2.5-4.2`, `results/P45B_discharge_3.0-4.2` |
| Fig. 7 | `figures/fig_bias_modes.py 7` | `results/P45B_discharge_2.94-4.2`, `results/P45B_charge_2.94-4.2` |
| Fig. S1 | `figures/figS1_peak_alignment.py` | the same two analyses as Fig. 7 |
| Fig. S2 | `figures/figS2_current_partition.py` | reference curves, `results/P45B_discharge_2.94-4.2`, `results/P45B_charge_2.94-4.2`, P45B data |
| Graphical abstract | `figures/graphical_abstract.py` | the analyses of Figs. 5-7, P45B data |
| Table S7 | `step3_tables.py` (`table_s7`) | `results/P45B_discharge_2.94-4.2`, `results/P45B_charge_2.94-4.2` and the two transition-point swap analyses (`..._charge-split`, `..._discharge-split`) |
| Table S8 | `step3_tables.py` (`table_s8`) | the three M50T analyses: uncorrected, IR-corrected (0.1 s) and `M50T_discharge_IR10s-corrected` |
| Tables S1-S6 | `step3_tables.py` (`table_s1` ... `table_s6`) | see the docstring of each function |
| Numbers in the text | `step5_text_numbers.py` | the results folder, the tables, and P45B and M50T data |

## 5. What is in the repository

```
paths.py                 where the data are (edit if needed)
step1_fit.py             fit every RPT of one or more analyses
step2_monte_carlo.py     half-cell offset Monte Carlo
step3_tables.py          Tables S1-S8
step4_figures.py         all data figures
step5_text_numbers.py    numbers quoted in the text
phantom_dma/
  analyses.py            every analysis in the paper and its settings
  data.py                read the raw data files; pulse resistance; IR correction
  model.py               full-cell voltage from the half-cell curves
  fit.py                 objective function and optimiser
  uncertainty.py         Monte Carlo and the paired uncertainty of a difference
  peaks.py               peak positions in dz/dV (Table S1, Fig. S1)
figures/                 one script per figure
reference_curves/        half-cell reference curves
results/                 fitted results for every analysis (see below)
```

### Results folder

One folder per analysis, for example `results/P45B_discharge_2.5-4.2/`:

| File | Content |
|---|---|
| `cell1_summary.csv`, `cell2_summary.csv`, `summary.csv` | One row per RPT: fitted parameters, electrode and lithium capacities, degradation modes, fit quality |
| `fits/cell1_RPT_000.npz`, ... | Measured and fitted voltage on the full-cell SOC grid, fitted parameters and transition point, for each RPT |
| `cell1_mc.csv`, `cell2_mc.csv` | Every Monte Carlo draw |
| `uncertainty.csv` | Mean of the two cells, half their range, Monte Carlo standard deviation and the two combined |

The columns of `summary.csv` are described in `data_dictionary.csv` of the Zenodo record
(file `DMA_fit_results.csv` there holds the same numbers).

Names used in the code:

- `cell1`, `cell2`: the two cells of each type. For the LG M50T they are the cells called
  D and E in Kirkaldy et al.
- RPT: for the P45B, the number of ageing cycles completed (0 to 500); for the LG M50T, the
  test index of Kirkaldy et al. The paper uses the even tests 0 to 14;
  `results/M50T_discharge_uncorrected` also holds the odd tests and test 15.
- `nu_NE`, `nu_PE` (stretch), `sigma_NE`, `sigma_PE` (shift), `phi_Si` (silicon share of the
  negative electrode): the five fitted parameters.
- `LAM_PE_pct`, `LAM_NE_pct`, `LLI_pct`, `LAM_Si_pct`, `LAM_Gr_pct`: degradation modes in %
  relative to RPT 0. The silicon and graphite values are written LAM\* in the paper.

## 6. Notes

- The optimiser (differential evolution) uses fixed random seeds, so a rerun gives the same
  numbers.
- Figs. 2-4 also write the numbers they plot, as CSV files next to the figure.
- Figs. 3 and 4 of the paper were drawn with matplotlib 3.10, the others with 3.11.
  Another version can place text a pixel differently; the data drawn are the same.
- Every analysis, the window screens included, uses ten optimiser trials per RPT and the
  50-draw Monte Carlo, except `M50T_discharge_IR10s-corrected` (the 10 s resistance
  sensitivity test of Section 4.4 and Table S8), which has nominal fits only.
- The two transition-point swap analyses (`P45B_discharge_2.94-4.2_charge-split`,
  `P45B_charge_2.94-4.2_discharge-split`) take the transition point of each RPT from the
  fitted results of the other branch, so fit `P45B_charge_2.94-4.2` and
  `P45B_discharge_2.94-4.2` first when refitting from scratch.
- For windows that end below 4.2 V, the full-cell capacity is the capacity passed inside
  the window. No analysis in the paper uses such a window.

## 7. Cite

If you use this code, please cite the paper above, this code (see `CITATION.cff`), and the
datasets you use: the P45B record (doi:10.5281/zenodo.23538218) and, for the LG M50T,
Kirkaldy et al., J. Power Sources 603 (2024) 234185 and doi:10.5281/zenodo.10637534.

## 8. Licence and contact

Code: BSD 3-Clause licence (see `LICENSE`). Data files keep the licences of their sources
(see `reference_curves/README.md`).

Contact: Mohammed Asheruddin N, Department of Mechanical Engineering, Imperial College London,
m.nazeeruddin@imperial.ac.uk.

## Funding

This work was supported by the Faraday Institution (grant numbers FIRG084 and FIRG095) and by
Innovate UK through the Faraday Battery Challenge (project number 10081877).
