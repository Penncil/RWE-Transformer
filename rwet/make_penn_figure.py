"""Penn Medicine ADRD figure on a single cohort: all arms, one drug, one panel.

Companion to make_funnel_grid.py (MIMIC-IV) and drawn with the same routines, so
the two datasets are the same plot. The three population columns come from
results/adrd_nco_<DRUG>.csv, produced by run_adrd_nco.py; the ITE column is
computed here with the DR-learner used for MIMIC-IV.

Unlike the five-drug assembled figure this holds the cohort, the control panel
and the estimator fixed across rows, so the differences between rows are
attributable to the adjustment set.

    DRUG=40790 python make_penn_figure.py

Output: figures/penn_grid_<DRUG>.pdf and Image RWE-GPT/results/penn_grid.pdf
"""
from __future__ import annotations

import os

from rwet import paths
import re
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rwet import plotstyle as ps  # noqa: E402
from rwet.baselines import fit_bcauss, fit_causalegm  # noqa: E402
from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402
from rwet.ite_core import dr_learner  # noqa: E402
from rwet.make_funnel import ite_panel  # noqa: E402

DRUG = os.environ.get("DRUG", "40790")
SRC = (str(paths.AD_DATA) +
       rf"\data_for_bingyu_300\{DRUG}_data.csv")
OUT = str(paths.RESULTS)
FIGS = str(paths.FIGURES)
FIGDIR = str(paths.FIGURES)
FIG_W, ROW_H = 13.0, 2.20
N_COV, N_PC, MIN_EV = 30, 15, 25

ROWS = [("X", "Baseline"), ("BCAUSS", "BCAUSS"), ("CausalEGM", "CausalEGM"),
        ("Z", "Ours\n(representation)"), ("XZ", "Ours\n(+ covariates)")]
COLS = ["ATT", "ATE", "ATC", "ITE"]


def _std(M):
    M = np.asarray(M, float)
    s = M.std(0)
    s[s == 0] = 1.0
    return (M - M.mean(0)) / s


def build():
    """Identical construction to run_adrd_nco.py, so the arms match its output."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    d = pd.read_csv(SRC)
    V = [c for c in d.columns if re.fullmatch(r"V\d+", c)]
    val = [c for c in d.columns if c.startswith("outcome_") and c.endswith("_value")]
    tim = [c for c in d.columns if c.startswith("outcome_") and c.endswith("_time")]
    skip = set(V + val + tim + ["Unnamed: 0", "ID", "treatment", "index_date"])
    cov = [c for c in d.columns if c not in skip
           and pd.api.types.is_numeric_dtype(d[c])]
    A = d["treatment"].to_numpy(float)
    Xall = d[cov].fillna(d[cov].median()).to_numpy(float)
    sel = LogisticRegressionCV(Cs=8, cv=5, penalty="l1", solver="saga",
                               scoring="neg_log_loss", max_iter=4000,
                               random_state=0).fit(_std(Xall), A)
    w = np.abs(sel.coef_.ravel())
    keep = np.argsort(-w)[:N_COV]
    keep = keep[w[keep] > 0]
    X = Xall[:, keep]
    Z = PCA(n_components=N_PC, random_state=0).fit_transform(_std(d[V].to_numpy(float)))
    sets = {"X": X, "Z": Z, "XZ": np.hstack([_std(X), _std(Z)])}
    return d, A, sets, val


def compute_ite():
    # Prefer the corrected column. At its default 300 epochs BCAUSS saturates on
    # this cohort -- 54-71% of predicted risks sit at exactly 0 or 1, so the
    # individual effects are +/-1 and the spread measures overfitting rather
    # than the method. fix_adrd_ite.py selects training length out of sample,
    # by the same rule for both neural arms.
    fixed = os.path.join(OUT, f"adrd_ite_{DRUG}_fixed.csv")
    if os.path.exists(fixed):
        print(f"using out-of-sample-selected ITE: {fixed}")
        return pd.read_csv(fixed)
    p = os.path.join(OUT, f"adrd_ite_{DRUG}.csv")
    if os.path.exists(p):
        print(f"reusing {p}")
        return pd.read_csv(p)
    d, A, sets, val = build()
    rows = []
    for key, label in ROWS:
        n_ok = 0
        for c in val:
            name = c[len("outcome_"):-len("_value")]
            if name in ("ADRD", "AD"):
                continue
            v = d[c].to_numpy(float)
            ok = v >= 0
            y, a = v[ok], A[ok]
            if y.sum() < MIN_EV or y[a == 1].sum() < 1 or y[a == 0].sum() < 1:
                continue
            try:
                if key in ("BCAUSS", "CausalEGM"):
                    f = fit_bcauss if key == "BCAUSS" else fit_causalegm
                    y0, y1 = f(sets["X"][ok], a, y, seed=0)
                    tau = np.asarray(y1).ravel() - np.asarray(y0).ravel()
                else:
                    tau = dr_learner(sets[key][ok], a, y, seed=0)
            except Exception:
                continue
            if not np.all(np.isfinite(tau)):
                continue
            n_ok += 1
            rows.append(dict(method=key, cluster=name, n_events=float(y.sum()),
                             mean_abs=float(np.mean(np.abs(tau))),
                             sd=float(np.std(tau, ddof=1)),
                             mean=float(np.mean(tau)),
                             q10=float(np.quantile(tau, 0.10)),
                             q90=float(np.quantile(tau, 0.90))))
        print(f"  ITE {label.replace(chr(10), ' '):22s} {n_ok} controls", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(p, index=False)
    return t


def main():
    ps.use(FIG_W)
    os.makedirs(FIGS, exist_ok=True)
    nc = pd.read_csv(os.path.join(OUT, f"adrd_nco_{DRUG}.csv"))
    nc = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]
    ite = compute_ite()

    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(FIG_W, ROW_H * len(ROWS)))
    summary = []
    for i, (key, label) in enumerate(ROWS):
        for j, est in enumerate(COLS):
            ax = axes[i, j]
            title = est if i == 0 else None
            if est == "ITE":
                sub = ite[ite["method"] == key]
                if len(sub):
                    ite_panel(ax, sub, title=title)
                else:
                    ax.axis("off")
                continue
            s = nc[(nc["method"] == key) & (nc["estimand"] == est)]
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            ps.funnel(ax, s.logrr.values, s.se.values, mu=mu, sigma=sg,
                      ease=ease_from_null(mu, sg), title=title, ylab=(j == 0))
            summary.append(dict(method=key, estimand=est, mu=mu, sigma=sg,
                                ease=ease_from_null(mu, sg), k=k))
        ps.row_label(axes[i, 0], label, x=-0.92)

    fig.subplots_adjust(left=0.145, right=0.90, top=0.935, bottom=0.062,
                        hspace=0.60, wspace=0.42)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, f"penn_grid_{DRUG}_fits.csv"), index=False)
    for d_, name in ((FIGS, f"penn_grid_{DRUG}.png"),
                     (FIGDIR, "penn_grid.png")):
        os.makedirs(d_, exist_ok=True)
        p = os.path.join(d_, name)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"-> {p.replace('.png', '.pdf')}")
    plt.close(fig)
    print("\nEASE drawn:")
    print(sm.pivot(index="method", columns="estimand",
                   values="ease").round(3).to_string())
    print("\nfitted sigma (a value near zero means the null collapsed):")
    print(sm.pivot(index="method", columns="estimand",
                   values="sigma").round(4).to_string())


if __name__ == "__main__":
    main()
