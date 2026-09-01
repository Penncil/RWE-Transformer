"""Recompute the ADRD ITE column with training length chosen out of sample.

At its default 300 epochs BCAUSS saturates on this cohort: 54-71% of predicted
counterfactual risks sit at exactly 0 or 1, so the individual effects are +/-1
and the reported spread (0.398) measures overfitting rather than the method.
Training length is therefore selected per control by held-out log-loss, and the
same rule is applied to CausalEGM so neither arm is tuned harder than the other.
The DR-learner arms are unaffected -- they were already well behaved -- but are
recomputed here so the whole column comes from one script.

    DRUG=40790 python fix_adrd_ite.py

Output: results/adrd_ite_<DRUG>_fixed.csv
"""
from __future__ import annotations

import os

from rwet import paths
import re
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from rwet.baselines import fit_bcauss, fit_causalegm  # noqa: E402
from rwet.ite_core import dr_learner  # noqa: E402
from rwet.make_penn_figure import ROWS, build, MIN_EV  # noqa: E402

DRUG = os.environ.get("DRUG", "40790")
OUT = str(paths.RESULTS)
GRID = (10, 25, 50, 100, 300)
EPS = 1e-6


def logloss(y, p):
    p = np.clip(np.asarray(p, float).ravel(), EPS, 1 - EPS)
    y = np.asarray(y, float).ravel()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_selected(kind, X, a, y, seed=0):
    """Choose epochs on a held-out fifth, then refit on everything."""
    f = fit_bcauss if kind == "BCAUSS" else fit_causalegm
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                  stratify=np.stack([a, y > 0], 1).sum(1))
    except ValueError:
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)

    best, best_ep = np.inf, GRID[0]
    for ep in GRID:
        try:
            # Fitted on the training fold alone; X_eval asks the trained network
            # for counterfactual risks on the held-out fifth, whose outcomes and
            # covariates never entered the fit. Scoring the fitting rows instead
            # would make training length a function of training fit, which falls
            # monotonically in epochs and so selects nothing.
            y0, y1 = f(X[tr], a[tr], y[tr], seed=seed, epochs=ep, X_eval=X[te])
        except Exception:
            continue
        y0 = np.asarray(y0, float).ravel()
        y1 = np.asarray(y1, float).ravel()
        # predicted risk under each held-out patient's own observed arm
        p_te = np.where(a[te] == 1, y1, y0)
        ll = logloss(y[te], p_te)
        # penalise saturation explicitly: a fit pinned at 0/1 cannot be honest
        sat = np.mean((np.minimum(y0, y1) < EPS) | (np.maximum(y0, y1) > 1 - EPS))
        score = ll + 2.0 * sat
        if score < best:
            best, best_ep = score, ep
    y0, y1 = f(X, a, y, seed=seed, epochs=best_ep)
    tau = np.asarray(y1, float).ravel() - np.asarray(y0, float).ravel()
    return tau, best_ep


def main():
    d, A, sets, val = build()
    rows = []
    for key, label in ROWS:
        eps_used = []
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
                    tau, ep = fit_selected(key, sets["X"][ok], a, y)
                    eps_used.append(ep)
                else:
                    tau = dr_learner(sets[key][ok], a, y, seed=0)
            except Exception as exc:
                print(f"    {key} {name}: {exc}", flush=True)
                continue
            if not np.all(np.isfinite(tau)):
                continue
            rows.append(dict(method=key, cluster=name, n_events=float(y.sum()),
                             mean_abs=float(np.mean(np.abs(tau))),
                             sd=float(np.std(tau, ddof=1)),
                             mean=float(np.mean(tau)),
                             q10=float(np.quantile(tau, 0.10)),
                             q90=float(np.quantile(tau, 0.90))))
        note = (f"  epochs chosen: {sorted(set(eps_used))}" if eps_used else "")
        print(f"{label.replace(chr(10), ' '):24s} done{note}", flush=True)

    t = pd.DataFrame(rows)
    p = os.path.join(OUT, f"adrd_ite_{DRUG}_fixed.csv")
    t.to_csv(p, index=False)
    print(f"\n-> {p}\n")
    print(t.groupby("method")["sd"].agg(["count", "mean"]).round(4).to_string())


if __name__ == "__main__":
    main()
