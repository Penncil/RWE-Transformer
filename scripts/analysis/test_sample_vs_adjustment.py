"""Is the cov+rep advantage the adjustment, or the sample it selects?

Matching on the concatenated set keeps 94 to 163 patients where the covariates
keep 228 to 403. A small, well-balanced subset is exactly where negative
controls are easiest to centre, so the advantage may be the population rather
than the adjustment.

Direct test, no refitting required. For each control take the patients cov+rep
retained and estimate three ways on that same set:

  cov+rep weights   the arm as reported
  uniform weights   the retained sample with no adjustment at all
  covariate weights the covariates' own matching, restricted to those patients

If the uniform column already reaches the cov+rep EASE, the selection is doing
the work and the adjustment is adding little.

    DRUG=40790 python test_sample_vs_adjustment.py
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

from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402
from rwet.run_adrd_submitted_style import (  # noqa: E402
    _std, nn_match, poisson_logrr, propensity, subclass_weights)

B = str(paths.AD_DATA)
DRUG = os.environ.get("DRUG", "40790")
SRC = os.path.join(B, "data_for_bingyu_300", f"{DRUG}_data.csv")
OUT = str(paths.RESULTS)
MIN_EV = 25


def weights_for(e, a, est):
    if est == "ATT":
        return nn_match(e, a, 1, 3)
    if est == "ATC":
        return nn_match(e, a, 0, 1)
    return subclass_weights(e, a)


def main():
    d = pd.read_csv(SRC)
    V = [c for c in d.columns if re.fullmatch(r"V\d+", c)]
    val = [c for c in d.columns if c.startswith("outcome_") and c.endswith("_value")]
    tim = [c for c in d.columns if c.startswith("outcome_") and c.endswith("_time")]
    skip = set(V + val + tim + ["Unnamed: 0", "ID", "treatment", "index_date"])
    cov = [c for c in d.columns if c not in skip
           and pd.api.types.is_numeric_dtype(d[c])]
    A = d["treatment"].to_numpy(float)
    X = d[cov].fillna(d[cov].median()).to_numpy(float)
    Z = d[V].to_numpy(float)
    sets = {"X": X, "XZ": np.hstack([_std(X), _std(Z)])}
    e_all = {k: propensity(F, A) for k, F in sets.items()}

    panel = []
    for c in val:
        name = c[len("outcome_"):-len("_value")]
        if name in ("ADRD", "AD"):
            continue
        v = d[c].to_numpy(float)
        ok = v >= 0
        y = v[ok]
        if y.sum() < MIN_EV or y[A[ok] == 1].sum() < 1 or y[A[ok] == 0].sum() < 1:
            continue
        panel.append((name, c))
    print(f"drug {DRUG}: {len(d)} patients, panel {len(panel)} controls\n")

    rows = []
    for name, col in panel:
        v = d[col].to_numpy(float)
        ok = v >= 0
        y, a = v[ok], A[ok]
        eXZ, eX = e_all["XZ"][ok], e_all["X"][ok]
        for est in ("ATT", "ATE", "ATC"):
            wXZ = weights_for(eXZ, a, est)
            sel = wXZ > 0
            if sel.sum() < 20 or y[sel].sum() < 5:
                continue
            if len(np.unique(a[sel])) < 2:
                continue
            wX = weights_for(eX, a, est)
            variants = {
                "cov+rep weights": wXZ,
                "uniform on same patients": sel.astype(float),
                "covariate weights, same patients": np.where(sel, wX, 0.0),
            }
            for lab, w in variants.items():
                lr, se = poisson_logrr(y, a, w)
                rows.append(dict(variant=lab, estimand=est, cluster=name,
                                 n=int((w > 0).sum()), logrr=lr, se=se))

    nc = pd.DataFrame(rows)
    nc.to_csv(os.path.join(OUT, f"sample_vs_adjustment_{DRUG}.csv"), index=False)
    good = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]
    keep = (good.groupby(["estimand", "cluster"])["variant"].nunique()
            .rename("n").reset_index())
    keep = keep[keep["n"] == good["variant"].nunique()][["estimand", "cluster"]]
    good = good.merge(keep, on=["estimand", "cluster"])

    out = []
    for lab in good["variant"].unique():
        for est in ("ATT", "ATE", "ATC"):
            s = good[(good["variant"] == lab) & (good["estimand"] == est)]
            if len(s) < 3:
                continue
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            out.append(dict(variant=lab, estimand=est, mu=mu, sigma=sg,
                            ease=ease_from_null(mu, sg), k=k,
                            mean_n=float(s.n.mean())))
    ez = pd.DataFrame(out)
    ez.to_csv(os.path.join(OUT, f"sample_vs_adjustment_ease_{DRUG}.csv"),
              index=False)
    for c in ("ease", "mu", "k", "mean_n"):
        print(f"\n{c}\n")
        print(ez.pivot(index="variant", columns="estimand",
                       values=c).round(3).to_string())


if __name__ == "__main__":
    main()
