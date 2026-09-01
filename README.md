# RWE-Transformer

Code for *RWE-Transformer: a pretrained transformer model with negative control
outcomes for debiased real-world evidence*.

The model is a transformer **encoder** pre-trained with a next-visit objective
and a negative-control-outcome (NCO) penalty. Downstream, the encoder is frozen
and only a task-specific output layer is fitted; the learned representation is
used as an adjustment set alongside the measured covariates.

## No patient data is included

None of the cohorts can be redistributed, and nothing in this repository
contains a patient identifier.

| source | how to obtain it |
|---|---|
| MIMIC-IV v2.2 | PhysioNet, with credentialing and a signed Data Use Agreement |
| Penn Medicine ADRD | institutional data; not shareable |
| OneFlorida+ | institutional data; not shareable |
| OMOP vocabulary | download from Athena (`CONCEPT.csv` is required to resolve drug codes) |

This repository is **code only**. No tabular data ships with it, aggregate or
otherwise: every result table is derived from the cohorts above, and derived
tables stay with the cohorts rather than travelling separately. `.gitignore`
denies data formats outright, with no re-admitting exceptions.

## Layout

    rwet/                shared modules, importable as a package
      paths.py           every data root, resolved from the environment
      causal_core.py     cross-fitted nuisances, empirical null, EASE
      ite_core.py        DR-learner for individual effects
      baselines.py       BCAUSS and CausalEGM re-implementations
      plotstyle.py       the funnel-plot routine used by every figure
    scripts/train/       pre-training
    scripts/analysis/    cohort analyses
    scripts/figures/     figures and table bodies
                         (results/ and figures/ are created at run time)

## Setup

    python -m venv .venv && .venv/Scripts/activate      # Windows
    pip install -r requirements.txt
    pip install -e .

Point the code at your data:

    set RWET_DATA=D:/path/to/rwe-data                   # Windows
    export RWET_DATA=/path/to/rwe-data                  # Linux, macOS

`RWET_DATA` should contain `mimic/`, `ad-deid/` and `omop-vocabulary/`.
`RWET_WORK`, `RWET_RESULTS` and `RWET_FIGURES` default to directories inside the
repository and rarely need setting.

## Reproducing the results

Because no result files are distributed, reproduction starts from the cohorts.
Run the analyses first; each writes into `results/`, which the figure scripts
then read. Nothing under `scripts/figures/` will run against an empty
`results/` directory.

Pre-train, then analyse:

    python scripts/train/run_h64e3.py                    # 4 seeds, ~20 min each on one RTX 4090
    python scripts/analysis/run_baselines_mimic.py       # BCAUSS and CausalEGM comparators
    python scripts/analysis/eval_full_insample.py        # ARM=F64; MIMIC-IV EASE
    python scripts/analysis/make_ite_grid.py             # individual effects
    python scripts/analysis/run_adrd_nco.py              # DRUG=40790
    python scripts/analysis/fix_adrd_ite.py              # DRUG=40790
    python scripts/analysis/run_penn_active_comparator.py

Then draw:

    python scripts/figures/make_funnel_grid.py     # Figure 4, MIMIC-IV funnels
    python scripts/figures/make_mimic_tables.py    # Tables 2 and 3
    python scripts/figures/make_fig5.py            # Figure 5, ADRD
    python scripts/figures/make_adrd_table.py      # Table 4
    python scripts/figures/make_uf_figure.py       # Figure 8, OneFlorida+

Figure 8 is a special case. The OneFlorida+ per-control estimates were never
written to disk, so that panel cannot be recomputed from any surviving artefact;
`scripts/analysis/recover_uf_funnel.py` reconstructs the scatter from the
published calibration images, and `make_uf_nulls.py` lifts the fitted nulls out
of the calibrated summary files. Both are documented in their own docstrings and
in the paper's caption for that figure.

## Two things to know before reading the numbers

**The headline MIMIC-IV tables are in-sample.** They reproduce the submitted
protocol, in which the penalty trains on all nineteen admitted negative controls
and EASE is then scored on those same nineteen. `rwet/nco_split_full.py` is that
no-hold-out arm, `scripts/analysis/eval_full_insample.py` scores it, and the
paper labels the values in-sample by construction. `rwet/nco_split.py` holds the
disjoint objective/validation/test split behind the held-out cross-check
reported alongside them.

**The individual-effect column is a diagnostic, not a win.** On both cohorts
CausalEGM attains a lower spurious individual-effect spread than the learned
representation, and on MIMIC-IV the representation is also worse than the same
architecture trained without the penalty. The method is for population-level
confounding control; it is not claimed to improve individual-level estimation.

## The negative-control penalty

The penalty is the mean squared partial correlation between NCO residuals and
treatment residuals. `torch.corrcoef` treats **rows** as variables, so the
residual matrix must be transposed before the correlation is taken:

    stacked = torch.hstack([res_W, res_T]).T
    rho = torch.corrcoef(stacked)[:d_w, d_w:]
    penalty = rho.pow(2).mean()

Without the transpose the statistic correlates patients with one another and is
flat in the NCO-treatment dependence it is meant to measure. Every result in the
paper uses the transposed form. `scripts/train/stage3b_sweep_train.py` takes
`--penalty {corrected,asreleased}` so the two can be compared directly.

## Citation

Please cite the paper. A `CITATION.cff` will be added on acceptance.

## License

MIT, see `LICENSE`. This covers the code in this repository only. It grants no
rights over MIMIC-IV, the Penn Medicine ADRD extract or the OneFlorida+ extract,
each of which carries its own agreement and none of which is distributed here.
