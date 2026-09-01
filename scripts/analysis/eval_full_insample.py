"""In-sample EASE for the no-hold-out arm -- the submitted protocol, re-derived cohort.

The penalty for the C*L10FULL models trained on all 19 admitted negative
controls, with no objective/validation split, exactly as the submitted pipeline
did with its 40. EASE is then computed on those same 19. Every number this
produces is IN-SAMPLE and must be labelled so.

Comparators (unadjusted, covariates, BCAUSS, CausalEGM) are scored on the same
19 controls, as the submission also did.

    python eval_full_insample.py

Output: results/full_insample_nco.csv, results/full_insample_ease.csv
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
from rwet.make_funnel import logrr_all  # noqa: E402

OUT = str(paths.RESULTS)
WORK = str(paths.WORK)
# (tag suffix, Z0 comparison tag, label) -- "debias" is the ORIGINAL model from
# stage3_train.py: penalty over every cluster, no split, hidden 64, 5 epochs.
# It is the submitted protocol's own network, so it is scored alongside the
# retrained hidden-128 arm rather than instead of it.
#
# ARM selects which no-hold-out arm to score:
#   FULL  hidden 128, 2 epochs, seeds 201-203, lambda=0 companion at hidden 128
#   F64   hidden  64, 3 epochs, seeds 201-204, lambda=0 companion at hidden 64
# The companion width must match the treated arm, or the "w/o debias" row
# ablates a different architecture from the one above it.
ARM = os.environ.get("ARM", "FULL").upper()
if ARM == "F64":
    SEEDS = [201, 202, 203, 204]
    ZFMT, Z0FMT = "C{s}L10F64", "C{s}L0F64"
    ARCH = "hidden 64, 3 epochs"
    LEGACY = []
else:
    SEEDS = [201, 202, 203]
    ZFMT, Z0FMT = "C{s}L10FULL", "C{s}L0"
    ARCH = "hidden 128, 2 epochs"
    LEGACY = [("debias", "nodebias", "submitted arch (h64, 5 ep), all NCOs")]
LABEL = {"Unadjusted": "Baseline (unadjusted)", "X": "Baseline (covariates)",
         "BCAUSS": "BCAUSS", "CausalEGM": "CausalEGM",
         "Z0": "Ours w/o debias", "Z": "Ours (representation)",
         "XZ": "Ours (with covariates)"}
ROWS = ["Unadjusted", "X", "BCAUSS", "CausalEGM", "Z0", "Z", "XZ"]
# BCAUSS and CausalEGM never see the control panel in training, so their
# per-control estimates do not depend on the pre-training arm and are reused
# across seeds. They are folded in here so the common-panel intersection covers
# them too -- otherwise the neural comparators would be scored on a different
# set of controls from the rows they are being compared against.
BASELINE_NCO = "funnel_baselines_nco.csv"
BASELINE_SRC = "mimic_with_baselines.csv"
BASELINES = ("BCAUSS", "CausalEGM")
FN_NCO = f"full_insample_nco_{ARM}.csv"
FN_EASE = f"full_insample_ease_{ARM}.csv"
CRUDE_LOGRR = np.log(1.801)      # unadjusted RR on this cohort


def baseline_eff():
    """Central log risk ratios for the two neural comparators.

    Read from results/mimic_with_baselines.csv rather than transcribed. These
    are the only rows in the table not recomputed by this script -- the
    comparators do not depend on the pre-training arm, so refitting them once
    per seed would return the same number four times -- and reading them keeps
    that single source of truth visible instead of frozen into a literal.
    """
    p = os.path.join(OUT, BASELINE_SRC)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; it carries the comparator effect "
                         "estimates and there is no fallback for them")
    b = pd.read_csv(p).set_index("method")
    return {m: float(b.loc[m, "logRR"]) for m in BASELINES}


def panel_names():
    from rwet.nco_split_full import three_way
    obj, _v, _t, clusters, _W = three_way()
    return [clusters[j] for j in obj]


def one_seed(seed, panel, ztag=None, z0tag=None):
    os.environ["ZTAG"] = ztag or ZFMT.format(s=seed)
    os.environ["Z0TAG"] = z0tag or Z0FMT.format(s=seed)
    os.environ["PANEL"] = "all"
    for m in ("rwet.make_funnel_corrected",):
        sys.modules.pop(m, None)
    from rwet.make_funnel_corrected import build
    sets, A, Y, ncoW, _p = build()

    recs = []
    for key, F in sets.items():
        for c in panel:
            if c not in ncoW.columns:
                continue
            w = ncoW[c].to_numpy(float)
            if not (1 <= w.sum() <= len(w) - 1):
                continue
            try:
                r = logrr_all(F, A, w)
            except Exception:
                continue
            for est, (e_, s_) in r.items():
                recs.append(dict(seed=seed, method=key, estimand=est,
                                 cluster=c, n_events=int(w.sum()),
                                 logrr=e_, se=s_))
        print(f"    {key}: done", flush=True)

    # the central estimate, for the distance-closed column
    eff = {}
    for key, F in sets.items():
        try:
            eff[key] = logrr_all(F, A, Y)
        except Exception:
            eff[key] = {}
    return pd.DataFrame(recs), eff


def main():
    b_eff = baseline_eff()
    panel = panel_names()
    print(f"in-sample panel: {len(panel)} controls (no hold-out)\n")

    nco, effs = [], {}
    for s in SEEDS:
        if not os.path.exists(os.path.join(
                WORK, "anchor_h48_control_" + ZFMT.format(s=s) + ".csv")):
            print(f"  seed {s}: not extracted yet, skipping")
            continue
        print(f"  seed {s}:")
        d, eff = one_seed(s, panel)
        nco.append(d)
        effs[s] = eff

    for ztag, z0tag, note in LEGACY:
        if not os.path.exists(os.path.join(WORK, f"anchor_h48_control_{ztag}.csv")):
            print(f"  {ztag}: not extracted yet, skipping")
            continue
        print(f"  {ztag} ({note}):")
        d, eff = one_seed(0, panel, ztag=ztag, z0tag=z0tag)
        d["seed"] = 0
        nco.append(d)
        effs[0] = eff

    if not nco:
        print("nothing to evaluate")
        return
    nco = pd.concat(nco, ignore_index=True)

    bl = os.path.join(OUT, BASELINE_NCO)
    if os.path.exists(bl):
        b = pd.read_csv(bl)
        b = b[b["method"].isin(BASELINES)][
            ["method", "estimand", "cluster", "logrr", "se"]]
        # same rows attached to every seed: deterministic given the cohort
        nco = pd.concat([nco] + [b.assign(seed=s) for s in nco["seed"].unique()],
                        ignore_index=True)
        print(f"  merged {b['method'].nunique()} neural comparators from {BASELINE_NCO}")
    else:
        print(f"  NOTE: {bl} missing -- BCAUSS/CausalEGM rows omitted")

    nco.to_csv(os.path.join(OUT, FN_NCO), index=False)

    # Like for like: within each seed and estimand, keep only the controls on
    # which EVERY method returned a usable estimate. Scoring some rows on an
    # easier subset than others is not a comparison, and the subsets do differ
    # -- the covariate fits diverge on a few of the rarest controls.
    ok = nco[np.isfinite(nco.logrr) & np.isfinite(nco.se)
             & (nco.se > 0) & (nco.se < 5)]
    n_meth = nco["method"].nunique()
    common = (ok.groupby(["seed", "estimand", "cluster"])["method"].nunique()
              .rename("n").reset_index())
    common = common[common["n"] == n_meth][["seed", "estimand", "cluster"]]
    ok = ok.merge(common, on=["seed", "estimand", "cluster"])
    print("\ncommon panel per estimand: "
          + ", ".join(f"{e} k={ok[ok.estimand == e].cluster.nunique()}"
                      for e in ("ATT", "ATE", "ATC")))

    rows = []
    for s in sorted(ok["seed"].unique()):
        for m in ROWS:
            for est in ("ATT", "ATE", "ATC"):
                d = ok[(ok["seed"] == s) & (ok["method"] == m)
                       & (ok["estimand"] == est)]
                if len(d) < 3:
                    continue
                mu, sig, k = fit_empirical_null(d.logrr.values, d.se.values)
                if m in BASELINES:
                    e = (b_eff[m], np.nan)
                else:
                    e = effs[s].get(m, {}).get(est, (np.nan, np.nan))
                closed = (100 * (1 - abs(e[0]) / abs(CRUDE_LOGRR))
                          if np.isfinite(e[0]) else np.nan)
                rows.append(dict(seed=s, method=m, label=LABEL[m], estimand=est,
                                 mu=mu, sigma=sig, ease=ease_from_null(mu, sig),
                                 k=k, logrr=e[0], se=e[1], closed=closed))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, FN_EASE), index=False)

    for group, sel, head in (
            ("retrained", out["seed"] > 0,
             f"IN-SAMPLE EASE, {ARCH}, mean over "
             f"{out[out['seed'] > 0]['seed'].nunique()} seeds (sd)"),
            ("legacy", out["seed"] == 0,
             "IN-SAMPLE EASE, submitted architecture (h64, 5 epochs), 1 model")):
        g = out[sel]
        if not len(g):
            continue
        print(f"\n=== {head} ===")
        for est in ("ATT", "ATE", "ATC"):
            print(f"\n  {est}")
            for m in ROWS:
                d = g[(g["method"] == m) & (g["estimand"] == est)]
                if not len(d):
                    continue
                sd = d["ease"].std() if len(d) > 1 else 0.0
                print(f"    {LABEL[m]:26s} EASE {d['ease'].mean():.3f} ({sd:.3f})"
                      f"   closed {d['closed'].mean():5.1f}%   k={int(d['k'].mean())}")
    print(f"\n-> {os.path.join(OUT, FN_EASE)}")


if __name__ == "__main__":
    main()
