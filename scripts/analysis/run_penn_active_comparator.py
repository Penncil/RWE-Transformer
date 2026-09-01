"""P-1: the active-comparator sensitivity analysis Reviewer 2 asked for.

The submitted Penn Medicine analysis compares metoprolol initiators against users
of folic acid and vitamin B12. The referee's objection is that this comparator is
not inert in a cognitively impaired population -- supplementation is often started
in response to anaemia, malnutrition or a reversible-cause workup prompted by the
cognitive complaint itself -- so confounding by indication acts in the comparator
arm as well as the treated one.

This rebuilds the contrast with an active comparator. Treated patients are the
metoprolol initiators; the comparator arm is the treated arm of a second
antihypertensive, so both arms carry a cardiovascular indication. Patients who
initiated both are dropped, since they cannot serve as their own control.

The comparator drug is identified from the cohort tables rather than from a code
dictionary: its treated arm is 95.4% hypertensive against 85.9% for metoprolol,
with a lower atrial-fibrillation share (25.3% against 38.2%), consistent with a
different antihypertensive class. The RxNorm ingredient identifier is reported so
the authors can confirm the drug name.

What this analysis can and cannot deliver. The per-drug tables carry the negative
control outcomes but not progression to Alzheimer's disease, so the systematic
error diagnostic can be recomputed under the active comparator and the treatment
effect cannot. Since the referee's concern is residual confounding, and EASE is
the quantity that measures it, the diagnostic is the part of the sensitivity
analysis that bears on the objection.

    COMPARATOR=17767 python run_penn_active_comparator.py

Output: results/penn_active_{TREAT}_vs_{COMPARATOR}.csv and _ease.csv
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402

from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402
from rwet.make_funnel import logrr_all  # noqa: E402

B = (str(paths.AD_DATA) +
     r"\data_for_bingyu_300")
TREAT = os.environ.get("TREAT", "6918")          # metoprolol
COMPARATOR = os.environ.get("COMPARATOR", "17767")
OUT = str(paths.RESULTS)
MIN_EV = int(os.environ.get("MINEV", "25"))
N_COV = int(os.environ.get("NCOV", "30"))
N_PC = int(os.environ.get("NPC", "15"))
PRIMARY = {"ADRD", "AD"}


def _std(M):
    M = np.asarray(M, float)
    s = M.std(0)
    s[s == 0] = 1.0
    return (M - M.mean(0)) / s


def build():
    if COMPARATOR == "vitamin":
        # The submitted design, run through the identical estimator so that the
        # active-comparator result has something to be compared against.
        d = pd.read_csv(os.path.join(B, f"{TREAT}_data.csv"))
        print(f"vitamin comparator (as submitted): {TREAT}")
        print(f"  n = {len(d)}, treated = {int(d.treatment.sum())}, "
              f"comparator = {int((1 - d.treatment).sum())}")
    else:
        a = pd.read_csv(os.path.join(B, f"{TREAT}_data.csv"))
        b = pd.read_csv(os.path.join(B, f"{COMPARATOR}_data.csv"))
        at = a[a.treatment == 1].copy()
        bt = b[b.treatment == 1].copy()
        both = set(at.ID) & set(bt.ID)
        at = at[~at.ID.isin(both)]
        bt = bt[~bt.ID.isin(both)]
        at["treatment"], bt["treatment"] = 1.0, 0.0
        shared = [c for c in a.columns if c in set(b.columns)]
        d = pd.concat([at[shared], bt[shared]], ignore_index=True)
        print(f"active comparator: {TREAT} treated vs {COMPARATOR} treated")
        print(f"  {len(both)} patients initiated both and were dropped")
        print(f"  n = {len(d)}, treated = {int(d.treatment.sum())}, "
              f"comparator = {int((1 - d.treatment).sum())}")

    V = [c for c in d.columns if re.fullmatch(r"V\d+", c)]
    val = [c for c in d.columns if c.startswith("outcome_")
           and c.endswith("_value")]
    tim = [c for c in d.columns if c.startswith("outcome_")
           and c.endswith("_time")]
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
    Zfull = d[V].to_numpy(float)
    pca = PCA(n_components=N_PC, random_state=0).fit(_std(Zfull))
    Z = pca.transform(_std(Zfull))
    print(f"  LASSO screen: {int((w > 0).sum())} of {len(cov)} retained, "
          f"using top {len(keep)}; representation {len(V)} -> {N_PC} PCs")
    sets = {"Unadjusted": np.zeros((len(A), 1)), "X": X, "Z": Z,
            "XZ": np.hstack([_std(X), _std(Z)])}
    return d, A, sets, val


def main():
    os.makedirs(OUT, exist_ok=True)
    d, A, sets, val = build()
    tag = f"{TREAT}_vs_{COMPARATOR}"

    panel = []
    for c in val:
        name = c[len("outcome_"):-len("_value")]
        if name in PRIMARY:
            continue
        v = d[c].to_numpy(float)
        ok = v >= 0
        y = v[ok]
        if y.sum() < MIN_EV or y[A[ok] == 1].sum() < 1 or y[A[ok] == 0].sum() < 1:
            continue
        panel.append((name, c))
    print(f"panel: {len(panel)} controls with >= {MIN_EV} events\n", flush=True)

    rows = []
    for name, col in panel:
        v = d[col].to_numpy(float)
        ok = v >= 0
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
        print(f"  {name}: {int(y.sum())} events / {int(ok.sum())} evaluable",
              flush=True)

    nc = pd.DataFrame(rows)
    nc.to_csv(os.path.join(OUT, f"penn_active_{tag}.csv"), index=False)
    good = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0)
              & (nc.se < 5)]
    keep = (good.groupby(["estimand", "cluster"])["method"].nunique()
            .rename("n").reset_index())
    keep = keep[keep["n"] == good["method"].nunique()][["estimand", "cluster"]]
    good = good.merge(keep, on=["estimand", "cluster"])

    out = []
    for m in nc["method"].unique():
        for e in ("ATT", "ATE", "ATC"):
            s = good[(good["method"] == m) & (good["estimand"] == e)]
            if len(s) < 3:
                continue
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            out.append(dict(method=m, estimand=e, mu=mu, sigma=sg,
                            ease=ease_from_null(mu, sg), k=k,
                            med_se=float(s.se.median())))
    ez = pd.DataFrame(out)
    ez.to_csv(os.path.join(OUT, f"penn_active_{tag}_ease.csv"), index=False)
    for c in ("ease", "mu", "sigma", "k", "med_se"):
        print(f"\n{c}\n")
        print(ez.pivot(index="method", columns="estimand",
                       values=c).round(3).to_string())


if __name__ == "__main__":
    main()
