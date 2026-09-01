"""
BCAUSS and CausalEGM on the re-derived MIMIC-IV cohort.

Both are reimplementations (improved/baselines.py) validated against the
simulated benchmark first: on known truth their errors are 0.163 and 0.068
against 0.218 for the unadjusted contrast, so both improve on doing nothing and
are worth reporting.

Each is given the observed covariates, as the covariate-adjustment comparator is,
and produces counterfactual risks from which a log risk ratio follows. The same
procedure is applied to every held-out negative control, so EASE is computed for
these methods the same way as for the others.
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

from rwet.baselines import (bootstrap_refit, effect_from_counterfactuals,  # noqa: E402
                       fit_bcauss, fit_causalegm)
from rwet.causal_core import (_standardise, aipw_crossfit, ease_from_null,  # noqa: E402
                         fit_empirical_null, nco_null)

WORK = str(paths.WORK)
OUT = str(paths.RESULTS)
BENCH = np.log(0.80)
LM_H, OUTCOME_H = 48.0, 7 * 24.0
MIN_EV = 25


def main():
    coh = pd.read_csv(os.path.join(WORK, "cohort_corrected.csv"), index_col=0)
    cov = pd.read_csv(os.path.join(WORK, "cohort_corrected_covariates.csv"),
                      index_col=0)
    los_h = coh["los_d"] * 24
    at_risk = (los_h > LM_H) & ~((coh["t_af_h"].notna()) & (coh["t_af_h"] <= LM_H))
    lm = coh[at_risk].copy()
    lm["A"] = ((lm["t_dex_h"].notna()) & (lm["t_dex_h"] <= LM_H)).astype(float)
    lm["Y"] = ((lm["t_af_h"] > LM_H) & (lm["t_af_h"] <= OUTCOME_H)).astype(float)

    def load(tag):
        a = pd.read_csv(os.path.join(WORK, f"anchor_h48_treat_{tag}.csv"),
                        header=None)
        b = pd.read_csv(os.path.join(WORK, f"anchor_h48_control_{tag}.csv"),
                        header=None)
        z = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=0,
                                                                keep="first")
        return z.set_index(z[0].astype(np.int64)).drop(columns=[0])

    Z = load("C201L10")
    idx = lm.index.intersection(cov.index).intersection(Z.index)
    lm, cov, Z = lm.loc[idx], cov.loc[idx], Z.loc[idx]
    A = lm["A"].to_numpy(float)
    Y = lm["Y"].to_numpy(float)
    print(f"landmark sample {len(idx):,}  treated {int(A.sum()):,}  "
          f"events {int(Y.sum()):,}\n", flush=True)

    labcols = [c for c in cov.columns if c != "age"]
    lab = cov[labcols].apply(pd.to_numeric, errors="coerce")
    miss = lab.isna().astype(float); miss.columns = [f"{c}_m" for c in lab.columns]
    miss = miss.loc[:, miss.std() > 0]
    lab = lab.fillna(lab.median())
    age = pd.to_numeric(cov["age"], errors="coerce"); age = age.fillna(age.median())
    cats = [c for c in ("gender", "race") if c in lm.columns]
    dm = (pd.get_dummies(lm[cats].astype(str), drop_first=True).astype(float)
          if cats else pd.DataFrame(index=lm.index))
    X = pd.concat([lab, miss, age.rename("age"), dm], axis=1).to_numpy(float)
    Zm = Z.to_numpy(float)

    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    nco = pd.DataFrame(z["W"], index=z["ids"],
                       columns=list(z["clusters"])).reindex(idx).fillna(0)
    from rwet.nco_split_corrected import three_way
    _o, _v, test_idx, clusters, _W = three_way()
    keep = [clusters[j] for j in test_idx
            if clusters[j] in nco.columns
            and MIN_EV <= nco[clusters[j]].sum() <= len(nco) - MIN_EV]
    W = nco[keep].to_numpy(float)
    print(f"held-out control panel: {len(keep)} clusters\n", flush=True)

    p1, p0 = Y[A == 1].mean(), Y[A == 0].mean()
    crude = np.log(p1 / p0)
    den = crude - BENCH

    # Per-control estimates are retained as well as the fitted null, so that the
    # funnel figure and this table are drawn from one computation rather than two.
    per_nco = []

    def neural_ease(kind, F, n_boot=10):
        """Per-control estimate and standard error, refitting on each replicate.

        The empirical null weights each control by its own standard error, so a
        standard error that ignores model uncertainty would push the fitted
        dispersion, and therefore EASE, in an arbitrary direction. The refit is
        expensive but the alternative is not interpretable.
        """
        est, se = [], []
        for j in range(W.shape[1]):
            w = W[:, j]
            if w.sum() < MIN_EV:
                continue
            try:
                e, s, _ = bootstrap_refit(kind, F, A, w, n_boot=n_boot, seed=0)
                if np.isfinite(e) and np.isfinite(s) and s > 0:
                    est.append(e); se.append(s)
                    per_nco.append(dict(method=kind, cluster=keep[j],
                                        logrr=e, se=s))
            except Exception:
                continue
        mu, sg, k = fit_empirical_null(np.array(est), np.array(se))
        return ease_from_null(mu, sg), k

    rows = []
    # reference methods
    for name, F in (("Unadjusted", np.zeros((len(A), 1))),
                    ("X (covariates)", X), ("Z (representation)", Zm),
                    ("X + Z", np.hstack([_standardise(X), _standardise(Zm)]))):
        r = aipw_crossfit(F, A, Y)
        n = nco_null(F, A, W, min_events=MIN_EV, crossfit=True)
        rows.append(dict(method=name, logRR=r["logRR"][0], se=r["logRR"][1],
                         ease=n["ease"], n_nco=n["k"]))
        for e_, s_ in zip(n["estimates"], n["ses"]):
            per_nco.append(dict(method=name, cluster="", logrr=e_, se=s_))
        print(f"  {name:22s} RR {np.exp(r['logRR'][0]):.3f}  "
              f"EASE {n['ease']:.4f}", flush=True)

    # neural baselines, with standard errors from refitting on each resample
    for kind, name in (("bcauss", "BCAUSS"), ("causalegm", "CausalEGM")):
        est, se, nb = bootstrap_refit(kind, X, A, Y, n_boot=20, seed=0)
        y0, y1 = (fit_bcauss if kind == "bcauss" else fit_causalegm)(
            X, A, Y, seed=0)
        _, se_fixed = effect_from_counterfactuals(y0, y1, seed=0)
        ease, k = neural_ease(kind, X)
        rows.append(dict(method=name, logRR=est, se=se, ease=ease, n_nco=k,
                         se_predictions_fixed=se_fixed, n_boot=nb))
        print(f"  {name:22s} RR {np.exp(est):.3f}  SE {se:.4f} "
              f"(holding predictions fixed: {se_fixed:.4f})  "
              f"EASE {ease:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df["RR"] = np.exp(df.logRR)
    df["lo"] = np.exp(df.logRR - 1.96 * df.se)
    df["hi"] = np.exp(df.logRR + 1.96 * df.se)
    df["closed_pct"] = (crude - df.logRR) / den * 100
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "mimic_with_baselines.csv"), index=False)
    pd.DataFrame(per_nco).to_csv(
        os.path.join(OUT, "baselines_per_nco.csv"), index=False)

    print("\n" + "=" * 88)
    print(f"RE-DERIVED MIMIC-IV COHORT, 48 h LANDMARK "
          f"(unadjusted RR {np.exp(crude):.3f}; benchmark 0.80)")
    print("=" * 88)
    order = ["X (covariates)", "BCAUSS", "CausalEGM", "Z (representation)",
             "X + Z"]
    df["_o"] = df.method.map({m: i for i, m in enumerate(order)})
    print(df.sort_values("_o")[["method", "RR", "lo", "hi", "se", "ease",
                                "n_nco", "closed_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
