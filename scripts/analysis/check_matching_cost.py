"""What does each adjustment set cost in effective sample size?

Claim under test: our arms carry larger standard errors because matching on a
331-dimensional propensity retains a smaller effective sample, and EASE applies
no penalty for that. Both halves need checking -- the propensity is fitted by
LASSO, so its effective dimension may be far below 331, and the precision loss
may come from overlap rather than from dimension.

Reports, per arm and estimand: retained dimensions after LASSO, matched sample
size, effective sample size (sum w)^2 / sum w^2, and the median standard error.

    DRUG=40790 python check_matching_cost.py
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
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402

from rwet.run_adrd_submitted_style import (  # noqa: E402
    _std, nn_match, subclass_weights)

B = str(paths.AD_DATA)
DRUG = os.environ.get("DRUG", "40790")
SRC = os.path.join(B, "data_for_bingyu_300", f"{DRUG}_data.csv")
OUT = str(paths.RESULTS)
MIN_EV = 25


def ess(w):
    w = np.asarray(w, float)
    w = w[w > 0]
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0


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
    sets = {"X": X, "Z": Z, "XZ": np.hstack([_std(X), _std(Z)])}

    print(f"drug {DRUG}: {len(d)} patients, {int(A.sum())} treated\n")
    print(f"{'arm':4s} {'input dims':>11s} {'LASSO keeps':>12s} "
          f"{'ps range':>18s} {'overlap 0.1-0.9':>16s}")
    e_cache, dims = {}, {}
    for k, F in sets.items():
        m = LogisticRegressionCV(Cs=8, cv=5, penalty="l1", solver="saga",
                                 scoring="neg_log_loss", max_iter=4000,
                                 random_state=0).fit(_std(F), A)
        nz = int((np.abs(m.coef_.ravel()) > 1e-8).sum())
        e = np.clip(m.predict_proba(_std(F))[:, 1], 1e-6, 1 - 1e-6)
        e_cache[k], dims[k] = e, nz
        ov = float(np.mean((e > 0.1) & (e < 0.9)))
        print(f"{k:4s} {F.shape[1]:11d} {nz:12d} "
              f"{f'{e.min():.3f}-{e.max():.3f}':>18s} {ov:15.1%}")

    nc = pd.read_csv(os.path.join(OUT, f"penn_tuned_nco_{DRUG}.csv"))
    nc = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]

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

    print(f"\nmatched sample and effective sample size, averaged over "
          f"{len(panel)} controls\n")
    print(f"{'arm':4s} {'estimand':>9s} {'matched n':>10s} {'ESS':>8s} "
          f"{'ESS/n':>7s} {'med SE':>7s}")
    rows = []
    for k in ("X", "Z", "XZ"):
        for est in ("ATT", "ATE", "ATC"):
            ns, es = [], []
            for _name, col in panel:
                v = d[col].to_numpy(float)
                ok = v >= 0
                a, e = A[ok], e_cache[k][ok]
                w = (nn_match(e, a, 1, 3) if est == "ATT"
                     else nn_match(e, a, 0, 1) if est == "ATC"
                     else subclass_weights(e, a))
                ns.append(int((w > 0).sum()))
                es.append(ess(w))
            s = nc[(nc.method == k) & (nc.estimand == est)]
            med = float(s.se.median()) if len(s) else np.nan
            rows.append(dict(arm=k, estimand=est, dims=dims[k],
                             matched_n=np.mean(ns), ess=np.mean(es),
                             ratio=np.mean(es) / np.mean(ns), med_se=med))
            print(f"{k:4s} {est:>9s} {np.mean(ns):10.0f} {np.mean(es):8.0f} "
                  f"{np.mean(es)/np.mean(ns):7.2f} {med:7.3f}")
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT, f"matching_cost_{DRUG}.csv"), index=False)
    print(f"\n-> {os.path.join(OUT, f'matching_cost_{DRUG}.csv')}")


if __name__ == "__main__":
    main()
