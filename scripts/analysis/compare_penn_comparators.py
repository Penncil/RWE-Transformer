"""P-1, scored fairly: the two comparator designs on the controls they share.

The active-comparator cohort admits 12 negative controls and the submitted
vitamin-comparator cohort 15, so their EASE values are not directly comparable
as computed. This refits every empirical null on the intersection, so the two
designs differ by the comparator arm alone.

    python compare_penn_comparators.py
"""
from __future__ import annotations

import os

from rwet import paths
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402

OUT = str(paths.RESULTS)
DESIGNS = [("vitamin", "6918_vs_vitamin", "Folic acid / B12 (as submitted)"),
           ("active", "6918_vs_17767", "Active comparator (antihypertensive)")]
ARMS = [("Unadjusted", "Unadjusted"), ("X", "Covariates"),
        ("Z", "Representation"), ("XZ", "Covariates + representation")]


def load(tag):
    nc = pd.read_csv(os.path.join(OUT, f"penn_active_{tag}.csv"))
    return nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0)
              & (nc.se < 5)]


def main():
    frames = {k: load(t) for k, t, _ in DESIGNS}
    common = set.intersection(*(set(f.cluster.unique()) for f in frames.values()))
    print(f"controls: " + ", ".join(
        f"{k} {f.cluster.nunique()}" for k, f in frames.items())
        + f"; shared {len(common)}\n")

    rows = []
    for key, _tag, label in DESIGNS:
        f = frames[key]
        f = f[f.cluster.isin(common)]
        for arm, arm_label in ARMS:
            for est in ("ATT", "ATE", "ATC"):
                s = f[(f.method == arm) & (f.estimand == est)]
                if len(s) < 3:
                    continue
                mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
                rows.append(dict(design=label, arm=arm_label, estimand=est,
                                 ease=ease_from_null(mu, sg), mu=mu, sigma=sg,
                                 k=k, med_se=float(s.se.median())))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "penn_comparator_comparison.csv"), index=False)
    for c in ("ease", "mu", "sigma", "med_se"):
        print(f"\n{c}\n")
        print(r.pivot(index=["design", "arm"], columns="estimand",
                      values=c).reindex(columns=["ATT", "ATE", "ATC"])
              .round(3).to_string())
    print(f"\nk = {r.k.min()}--{r.k.max()} (shared panel)")


if __name__ == "__main__":
    main()
