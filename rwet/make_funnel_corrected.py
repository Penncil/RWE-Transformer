"""Figure 4, drawn entirely on the RE-DERIVED MIMIC-IV cohort.

The previous version of this figure was drawn on the cohort as submitted: it
existed to show what the submitted analysis looks like once the estimand-labelling
error is corrected. That job is done, and the paper's claims rest on the cohort
re-derived from charted rhythm data, so the figure should be drawn there too.

Everything here comes from the same objects that produce
Table~\\ref{tab:mimic-corrected}:

    cohort          work/cohort_corrected.csv, 48-hour landmark
    covariates      work/cohort_corrected_covariates.csv
    representation  work/anchor_h48_{treat,control}_C201L{0,10}.csv
    controls        work/cohort_corrected_nco.npz, validation + test panels

Twelve held-out negative controls (validation and test pooled), the same
convention the old figure used, rather than the six of the table -- a funnel with
six points shows nothing. Controls with fewer than MIN_EV events are dropped
because their standard errors are not usable.

    python make_funnel_corrected.py [--recompute]
"""
from __future__ import annotations

import os

from rwet import paths
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
from rwet.causal_core import _standardise, ease_from_null, fit_empirical_null  # noqa: E402
from rwet.ite_core import dr_learner  # noqa: E402
from rwet.make_funnel import ite_panel, logrr_all  # noqa: E402

WORK = str(paths.WORK)
OUT = str(paths.RESULTS)
FIGDIR = str(paths.FIGURES)

LM_H, OUTCOME_H = 48.0, 7 * 24.0
# MINEV=1 admits every control with at least one event, which is the panel the
# adjustment-set methods are scored on; the default of 25 is the figure's panel.
MIN_EV = int(os.environ.get("MINEV", "25"))
FIG_W = 13.0

# The five rows of the submitted figure, on the re-derived cohort. The combined
# adjustment set (X + Z) is reported in Table~\ref{tab:mimic-corrected} rather
# than drawn here, so that the figure matches the submitted comparison.
# The final row is the representation CONCATENATED WITH THE COVARIATES (X + Z,
# 192 dimensions), not the representation alone. Set OURS=Z to draw the
# representation on its own instead.
_OURS = os.environ.get("OURS", "XZ")
ROWS = [("X", "Baseline"),
        ("BCAUSS", "BCAUSS"),
        ("CausalEGM", "CausalEGM"),
        ("Z0", "Ours\nw/o debias"),
        (_OURS, "Ours")]


def build():
    """The landmark cohort exactly as run_baselines_mimic.py forms it."""
    coh = pd.read_csv(os.path.join(WORK, "cohort_corrected.csv"), index_col=0)
    cov = pd.read_csv(os.path.join(WORK, "cohort_corrected_covariates.csv"),
                      index_col=0)
    los_h = coh["los_d"] * 24
    at_risk = (los_h > LM_H) & ~(coh["t_af_h"].notna() & (coh["t_af_h"] <= LM_H))
    lm = coh[at_risk].copy()
    lm["A"] = (lm["t_dex_h"].notna() & (lm["t_dex_h"] <= LM_H)).astype(float)
    lm["Y"] = ((lm["t_af_h"] > LM_H) & (lm["t_af_h"] <= OUTCOME_H)).astype(float)

    def load(tag):
        a = pd.read_csv(os.path.join(WORK, f"anchor_h48_treat_{tag}.csv"),
                        header=None)
        b = pd.read_csv(os.path.join(WORK, f"anchor_h48_control_{tag}.csv"),
                        header=None)
        z = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=0,
                                                                keep="first")
        return z.set_index(z[0].astype(np.int64)).drop(columns=[0])

    # Tags are overridable so the same cohort assembly can be re-used for other
    # pre-training arms (e.g. the no-hold-out in-sample arm) without a second
    # copy of this code drifting away from it. Defaults are the reported arm.
    Z = load(os.environ.get("ZTAG", "C201L10"))
    Z0 = load(os.environ.get("Z0TAG", "C201L0"))
    idx = (lm.index.intersection(cov.index)
           .intersection(Z.index).intersection(Z0.index))
    lm, cov, Z, Z0 = lm.loc[idx], cov.loc[idx], Z.loc[idx], Z0.loc[idx]
    A, Y = lm["A"].to_numpy(float), lm["Y"].to_numpy(float)

    labcols = [c for c in cov.columns if c != "age"]
    lab = cov[labcols].apply(pd.to_numeric, errors="coerce")
    miss = lab.isna().astype(float)
    miss.columns = [f"{c}_m" for c in lab.columns]
    miss = miss.loc[:, miss.std() > 0]
    lab = lab.fillna(lab.median())
    age = pd.to_numeric(cov["age"], errors="coerce")
    age = age.fillna(age.median())
    cats = [c for c in ("gender", "race") if c in lm.columns]
    dm = (pd.get_dummies(lm[cats].astype(str), drop_first=True).astype(float)
          if cats else pd.DataFrame(index=lm.index))
    X = pd.concat([lab, miss, age.rename("age"), dm], axis=1).to_numpy(float)
    Zm, Z0m = Z.to_numpy(float), Z0.to_numpy(float)

    sets = {"Unadjusted": np.zeros((len(A), 1)), "X": X, "Z0": Z0m, "Z": Zm,
            "XZ": np.hstack([_standardise(X), _standardise(Zm)])}

    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    nco = pd.DataFrame(z["W"], index=z["ids"],
                       columns=list(z["clusters"])).reindex(idx).fillna(0)
    from rwet.nco_split_corrected import three_way
    obj_idx, val_idx, test_idx, clusters, _W = three_way()
    # PANEL=all   every usable control, as OHDSI empirical calibration does
    # PANEL=heldout  validation + test only, excluding the 7 the penalty trained on
    which = os.environ.get("PANEL", "all").lower()
    take = (list(obj_idx) + list(val_idx) + list(test_idx) if which == "all"
            else list(val_idx) + list(test_idx))
    panel = [clusters[j] for j in take
             if clusters[j] in nco.columns
             and MIN_EV <= nco[clusters[j]].sum() <= len(nco) - MIN_EV]
    print(f"landmark sample {len(idx):,}  treated {int(A.sum()):,}  "
          f"events {int(Y.sum()):,}")
    print(f"panel = {which}: {len(panel)} controls\n", flush=True)
    return sets, A, Y, nco[panel], panel


def main():
    ps.use(FIG_W)
    os.makedirs(OUT, exist_ok=True)
    sets, A, Y, ncoW, panel = build()
    Wm = ncoW.to_numpy(float)

    # Per-control estimates come from the same files the tables are built from,
    # so the figure and Table cannot drift apart. Every method is then drawn on
    # the SAME controls -- the intersection where all five returned a finite
    # estimate -- since scoring some rows on easier controls than others is not
    # a comparison.
    nc = pd.read_csv(os.path.join(OUT, "panel_sensitivity_nco.csv"))
    bl = os.path.join(OUT, "funnel_baselines_nco.csv")
    if os.path.exists(bl):
        nc = pd.concat([nc, pd.read_csv(bl)], ignore_index=True)
    else:
        print(f"  NOTE: {bl} missing -- run run_baselines_funnel.py", flush=True)
    nc["method"] = nc["method"].str.lower()
    nc = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0)]

    keys = [k.lower() for k, _ in ROWS]
    common = set.intersection(*[
        set(nc[(nc.method == k) & (nc.estimand == "ATE")].cluster)
        for k in keys]) if keys else set()
    nc = nc[nc.cluster.isin(common)]
    print(f"  common control panel: {len(common)} controls\n", flush=True)

    ite = pd.read_csv(os.path.join(OUT, "ite_corrected_per_nco.csv"))
    bl_ite = os.path.join(OUT, "ite_baselines_per_nco.csv")
    if os.path.exists(bl_ite):
        ite = pd.concat([ite, pd.read_csv(bl_ite)], ignore_index=True)
    ite["method"] = ite["method"].str.lower()
    ite_common = set.intersection(*[set(ite[ite.method == k].cluster)
                                    for k in keys])
    ite = ite[ite.cluster.isin(ite_common)]

    cols = ["ATE", "ATT", "ATC", "ITE"]
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(FIG_W, 2.20 * len(ROWS)))
    summary = []
    for i, (key, label) in enumerate(ROWS):
        for jcol, cname in enumerate(cols):
            ax = axes[i, jcol]
            title = cname if i == 0 else None
            if cname == "ITE":
                sub = (ite[ite["method"] == key.lower()] if len(ite) else ite)
                if len(sub):
                    ite_panel(ax, sub, title=title)
                else:
                    ax.axis("off")
                    if title:
                        ax.set_title(title, pad=6 * ps.scale())
                continue
            sub = nc[(nc["method"] == key.lower())
                     & (nc["estimand"] == cname)]
            mu, sg, k = fit_empirical_null(sub["logrr"].to_numpy(),
                                           sub["se"].to_numpy())
            ea = ease_from_null(mu, sg)
            ps.funnel(ax, sub["logrr"].to_numpy(), sub["se"].to_numpy(),
                      mu=mu, sigma=sg, ease=ea, title=title, ylab=(jcol == 0))
            summary.append(dict(method=key, label=label.replace("\n", " "),
                                estimand=cname, mu=mu, sigma=sg, ease=ea, k=k))
        ps.row_label(axes[i, 0], label, x=-0.95)

    fig.subplots_adjust(left=0.168, right=0.995, top=0.935, bottom=0.062,
                        hspace=0.60, wspace=0.42)
    dest = os.path.join(FIGDIR, "mimicease_rederived.png")
    fig.savefig(dest, bbox_inches="tight")
    fig.savefig(dest.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "funnel_corrected_ease.csv"), index=False)
    print(f"\n{sm.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
