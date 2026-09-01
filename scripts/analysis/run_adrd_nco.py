"""Negative-control estimates for the Penn Medicine ADRD cohort, all arms in one pipeline.

The submitted ADRD figure cannot be redrawn from its own outputs: the downstream
script writes every method's negative-control results to one path, so only the
last run survives. This recomputes them from the analysis table instead, with
the same estimator used for MIMIC-IV, so the two cohorts become comparable for
the first time.

Analysis table: data_for_bingyu_300/6918_data.csv
  1,295 patients (532 treated, 763 control)
  267 numeric covariates
  V1..V64  the stored representation -- byte-identical to
           AD_output_0917/6918_{treat,control}_featuresv2.csv on all 1,295 ids
  outcome_<name>_value  negative control indicator, -1 = not evaluable

Arms: unadjusted, covariates, BCAUSS, CausalEGM, representation, and
covariates+representation. The unpenalised encoder is NOT available -- only one
representation is stored for this cohort -- so that row cannot be produced.

    python run_adrd_nco.py

Output: results/adrd_nco.csv, results/adrd_ease.csv
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

from rwet.baselines import bootstrap_refit_all  # noqa: E402
from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402
from rwet.make_funnel import logrr_all  # noqa: E402

# 40790 is the exposure Figure 5 and Table 4 report, and the default the rest of
# the pipeline assumes; 6918 (metoprolol) is the exposure of the effect table and
# is run separately with DRUG=6918.
DRUG = os.environ.get("DRUG", "40790")
SRC = (str(paths.AD_DATA) +
       rf"\data_for_bingyu_300\{DRUG}_data.csv")
OUT = str(paths.RESULTS)
MIN_EV = int(os.environ.get("MINEV", "25"))
N_BOOT = int(os.environ.get("NBOOT", "5"))
N_COV = int(os.environ.get("NCOV", "30"))
N_PC = int(os.environ.get("NPC", "15"))
PRIMARY = {"ADRD", "AD"}


def _std(M):
    M = np.asarray(M, float)
    s = M.std(0)
    s[s == 0] = 1.0
    return (M - M.mean(0)) / s


def build():
    d = pd.read_csv(SRC)
    V = [c for c in d.columns if re.fullmatch(r"V\d+", c)]
    val = [c for c in d.columns if c.startswith("outcome_")
           and c.endswith("_value")]
    tim = [c for c in d.columns if c.startswith("outcome_")
           and c.endswith("_time")]
    skip = set(V + val + tim + ["Unnamed: 0", "ID", "treatment", "index_date"])
    cov = [c for c in d.columns if c not in skip]
    cov = [c for c in cov if pd.api.types.is_numeric_dtype(d[c])]

    A = d["treatment"].to_numpy(float)
    Xall = d[cov].fillna(d[cov].median()).to_numpy(float)

    # 267 covariates on the ~500 patients evaluable for a given control is far
    # past what the AIPW nuisance fit can carry: the pseudo-outcome mean for one
    # arm collapses onto the 1e-10 floor in logrr_all and the log ratio lands at
    # 20-23. The submitted R analysis did not use all of them either -- it fitted
    # the propensity with cv.glmnet(alpha=1), i.e. a LASSO. We reproduce that
    # here as an explicit screen, keeping the covariates the LASSO retains and
    # capping at N_COV so the adjustment set is the size the MIMIC-IV analysis
    # used (about thirty features on 13,649 patients).
    from sklearn.linear_model import LogisticRegressionCV
    Xs = _std(Xall)
    sel = LogisticRegressionCV(Cs=8, cv=5, penalty="l1", solver="saga",
                               scoring="neg_log_loss", max_iter=4000,
                               random_state=0).fit(Xs, A)
    w = np.abs(sel.coef_.ravel())
    keep = np.argsort(-w)[:N_COV]
    keep = keep[w[keep] > 0]
    X = Xall[:, keep]
    print(f"  LASSO screen: {int((w > 0).sum())} of {len(cov)} covariates "
          f"retained, using top {len(keep)}")
    print("  kept:", [cov[i] for i in keep[:10]])

    # The representation needs the same discipline. 64 dims on the ~500 patients
    # evaluable per control still breaks the nuisance fit on a third of the
    # panel. Reduce it to its leading principal components -- unsupervised, so
    # no treatment or outcome information leaks into the choice -- giving an
    # adjustment set of a size this cohort can carry, as the 128-dimensional
    # representation was on the 13,649-patient MIMIC-IV cohort.
    from sklearn.decomposition import PCA
    Zfull = d[V].to_numpy(float)
    pca = PCA(n_components=N_PC, random_state=0).fit(_std(Zfull))
    Z = pca.transform(_std(Zfull))
    print(f"  representation: {Zfull.shape[1]} dims -> {N_PC} components "
          f"({100 * pca.explained_variance_ratio_.sum():.0f}% of variance)")
    sets = {"Unadjusted": np.zeros((len(A), 1)), "X": X, "Z": Z,
            "XZ": np.hstack([_std(X), _std(Z)])}
    print(f"{len(d):,} patients, {int(A.sum())} treated, "
          f"{len(cov)} covariates, {len(V)} representation dims")
    return d, A, sets, val


def main():
    os.makedirs(OUT, exist_ok=True)
    d, A, sets, val = build()

    panel = []
    for c in val:
        name = c[len("outcome_"):-len("_value")]
        if name in PRIMARY:
            continue
        v = d[c].to_numpy(float)
        ok = v >= 0
        y = v[ok]
        if y.sum() < MIN_EV:
            continue
        if (y[A[ok] == 1].sum() < 1) or (y[A[ok] == 0].sum() < 1):
            continue
        panel.append((name, c))
    print(f"panel: {len(panel)} controls with >= {MIN_EV} events\n", flush=True)

    rows = []
    for name, col in panel:
        v = d[col].to_numpy(float)
        ok = v >= 0                      # -1 marks a patient not evaluable here
        y, a = v[ok], A[ok]
        for key, F in sets.items():
            try:
                r = logrr_all(F[ok], a, y)
            except Exception:
                continue
            for est, (e_, s_) in r.items():
                rows.append(dict(method=key, estimand=est, cluster=name,
                                 n_events=int(y.sum()), n=int(ok.sum()),
                                 logrr=e_, se=s_))
        for kind, label in (("bcauss", "BCAUSS"), ("causalegm", "CausalEGM")):
            try:
                est, se, _tau, _k = bootstrap_refit_all(
                    kind, sets["X"][ok], a, y, n_boot=N_BOOT, seed=0)
            except Exception as exc:
                print(f"    {label} failed on {name}: {exc}", flush=True)
                continue
            for e in ("ATT", "ATE", "ATC"):
                rows.append(dict(method=label, estimand=e, cluster=name,
                                 n_events=int(y.sum()), n=int(ok.sum()),
                                 logrr=est.get(e, np.nan),
                                 se=se.get(e, np.nan)))
        print(f"  {name}: {int(y.sum())} events / {int(ok.sum())} evaluable",
              flush=True)

    nc = pd.DataFrame(rows)
    nc.to_csv(os.path.join(OUT, f"adrd_nco_{DRUG}.csv"), index=False)

    # One fixed panel for every method -- the whole admitted set, as on
    # MIMIC-IV. Intersecting on "controls every method could estimate" is what
    # collapsed the panel to a third of its size before, and the survivors of
    # such an intersection are the controls that happened not to break, which is
    # a selected subset rather than a random one. Any control a method cannot
    # estimate is reported as a gap in n_used instead.
    good = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0)
              & (nc.se < 5)]
    n_panel = nc["cluster"].nunique()

    out = []
    for m in nc["method"].unique():
        for e in ("ATT", "ATE", "ATC"):
            s = good[(good["method"] == m) & (good["estimand"] == e)]
            if len(s) < 3:
                out.append(dict(method=m, estimand=e, mu=np.nan, sigma=np.nan,
                                ease=np.nan, k=len(s), n_panel=n_panel))
                continue
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            out.append(dict(method=m, estimand=e, mu=mu, sigma=sg,
                            ease=ease_from_null(mu, sg), k=k,
                            n_panel=n_panel))
    ez = pd.DataFrame(out)
    ez.to_csv(os.path.join(OUT, f"adrd_ease_{DRUG}.csv"), index=False)
    print(f"\nEASE (in-sample, fixed {n_panel}-control panel, drug {DRUG})\n")
    print(ez.pivot(index="method", columns="estimand",
                   values="ease").round(3).to_string())
    print(f"\ncontrols used of {n_panel}:")
    print(ez.pivot(index="method", columns="estimand", values="k").to_string())
    print(f"\n-> {os.path.join(OUT, f'adrd_ease_{DRUG}.csv')}")


if __name__ == "__main__":
    main()
