"""The submitted downstream analysis, reimplemented, with and without dimension reduction.

Faithful to analysis_ad.R rather than to the MIMIC-IV AIPW pipeline:

  propensity   LASSO logistic on the adjustment set        (cv.glmnet, alpha=1)
  ATT          nearest-neighbour matching, caliper 0.2, ratio 3
  ATE          subclassification into 6 strata
  ATC          nearest-neighbour matching, caliper 0.2, ratio 1
  estimate     glm(nco ~ treatment, poisson(log), weights = matching weights)

This is why the original tolerated 267 covariates and a 64-dimensional
representation without blowing up: it fits no outcome model. The AIPW pipeline
used for MIMIC-IV does, which is what forced the dimension reduction there. The
reduction is therefore an artefact of my estimator choice, not of the data, and
this script runs each adjustment set at full width AND reduced to 15 components
so the two can be compared directly.

    DRUG=40790 python run_adrd_submitted_style.py

Output: results/adrd_submitted_style_<DRUG>.csv  (per-control)
        results/adrd_submitted_style_ease_<DRUG>.csv
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
import statsmodels.api as sm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402

from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402

DRUG = os.environ.get("DRUG", "40790")
SRC = (str(paths.AD_DATA) +
       rf"\data_for_bingyu_300\{DRUG}_data.csv")
OUT = str(paths.RESULTS)
MIN_EV = 25
CALIPER = 0.2
N_SUBCLASS = 6
N_PC = 15


def _std(M):
    M = np.asarray(M, float)
    s = M.std(0)
    s[s == 0] = 1.0
    return (M - M.mean(0)) / s


def propensity(F, A, seed=0):
    m = LogisticRegressionCV(Cs=8, cv=5, penalty="l1", solver="saga",
                             scoring="neg_log_loss", max_iter=4000,
                             random_state=seed).fit(_std(F), A)
    e = m.predict_proba(_std(F))[:, 1]
    return np.clip(e, 1e-6, 1 - 1e-6)


def nn_match(e, A, target, ratio):
    """Greedy nearest-neighbour matching without replacement, on the linear
    propensity, with a caliper of 0.2 standard deviations -- MatchIt's default
    scale. Returns per-patient weights, zero for unmatched patients."""
    lp = np.log(e / (1 - e))
    cal = CALIPER * lp.std(ddof=1)
    tgt = np.where(A == target)[0]
    pool = list(np.where(A != target)[0])
    w = np.zeros(len(A))
    # process the target group in a fixed order, as MatchIt does by default
    for i in tgt[np.argsort(-lp[tgt])]:
        if not pool:
            break
        dist = np.abs(lp[pool] - lp[i])
        order = np.argsort(dist)[:ratio]
        picked = [pool[j] for j, dd in zip(order, dist[np.argsort(dist)][:ratio])
                  if dd <= cal]
        if not picked:
            continue
        w[i] = 1.0
        for j in picked:
            w[j] = 1.0 / len(picked)
            pool.remove(j)
    # rescale the comparison group to carry the same total weight
    tot_t, tot_c = w[A == target].sum(), w[A != target].sum()
    if tot_c > 0:
        w[A != target] *= tot_t / tot_c
    return w


def subclass_weights(e, A, k=N_SUBCLASS):
    """ATE subclassification: equal-frequency strata on the propensity, each
    arm reweighted to the stratum's share of the whole sample."""
    edges = np.quantile(e, np.linspace(0, 1, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    s = np.digitize(e, edges[1:-1])
    w = np.zeros(len(A))
    n, n1, n0 = len(A), (A == 1).sum(), (A == 0).sum()
    for g in np.unique(s):
        m = s == g
        ns, ns1, ns0 = m.sum(), (m & (A == 1)).sum(), (m & (A == 0)).sum()
        if ns1 == 0 or ns0 == 0:
            continue
        w[m & (A == 1)] = (ns / n) / (ns1 / n1)
        w[m & (A == 0)] = (ns / n) / (ns0 / n0)
    return w


def poisson_logrr(y, A, w):
    """glm(y ~ treatment, poisson(log), weights = w), as the R code does."""
    use = w > 0
    if use.sum() < 10 or y[use].sum() < 5:
        return np.nan, np.nan
    X = sm.add_constant(A[use].astype(float), has_constant="add")
    m = sm.GLM(y[use], X, family=sm.families.Poisson(),
               freq_weights=w[use]).fit()
    return float(m.params[1]), float(m.bse[1])


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
    Z15 = PCA(n_components=N_PC, random_state=0).fit_transform(_std(Z))
    XZ = np.hstack([_std(X), _std(Z)])
    XZ15 = PCA(n_components=N_PC, random_state=0).fit_transform(XZ)

    SETS = [("X", "covariates", X), ("Z", "representation (64)", Z),
            ("Z15", "representation (15 pc)", Z15),
            ("XZ", "cov + rep (full)", XZ), ("XZ15", "cov + rep (15 pc)", XZ15)]
    print(f"drug {DRUG}: {len(d)} patients, {int(A.sum())} treated, "
          f"{len(cov)} covariates, {len(V)} representation dims\n")

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
    print(f"panel: {len(panel)} controls\n", flush=True)

    rows = []
    for key, label, F in SETS:
        e_full = propensity(F, A)
        for name, col in panel:
            v = d[col].to_numpy(float)
            ok = v >= 0
            y, a, e = v[ok], A[ok], e_full[ok]
            for est in ("ATT", "ATE", "ATC"):
                if est == "ATT":
                    w = nn_match(e, a, 1, 3)
                elif est == "ATC":
                    w = nn_match(e, a, 0, 1)
                else:
                    w = subclass_weights(e, a)
                lr, se = poisson_logrr(y, a, w)
                rows.append(dict(method=key, label=label, estimand=est,
                                 cluster=name, dims=F.shape[1],
                                 n_events=int(y.sum()), logrr=lr, se=se))
        print(f"  {label:26s} dims={F.shape[1]:3d} done", flush=True)

    nc = pd.DataFrame(rows)
    nc.to_csv(os.path.join(OUT, f"adrd_submitted_style_{DRUG}.csv"), index=False)

    out = []
    good = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]
    for key, label, F in SETS:
        for est in ("ATT", "ATE", "ATC"):
            s = good[(good["method"] == key) & (good["estimand"] == est)]
            if len(s) < 3:
                out.append(dict(method=key, label=label, estimand=est,
                                mu=np.nan, sigma=np.nan, ease=np.nan, k=len(s)))
                continue
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            out.append(dict(method=key, label=label, estimand=est, mu=mu,
                            sigma=sg, ease=ease_from_null(mu, sg), k=k))
    ez = pd.DataFrame(out)
    ez.to_csv(os.path.join(OUT, f"adrd_submitted_style_ease_{DRUG}.csv"),
              index=False)
    print("\nEASE, submitted-style downstream\n")
    print(ez.pivot(index="label", columns="estimand",
                   values="ease").round(3).to_string())
    print("\ncontrols used\n")
    print(ez.pivot(index="label", columns="estimand", values="k").to_string())
    print("\nfitted sigma\n")
    print(ez.pivot(index="label", columns="estimand",
                   values="sigma").round(4).to_string())


if __name__ == "__main__":
    main()
