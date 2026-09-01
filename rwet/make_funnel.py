"""
Redraw the MIMIC-IV negative-control funnel plot, corrected.

Two things are fixed relative to the submitted figure.

1. SCALE. Every panel now plots a genuine log risk ratio for its target
   population, so the "Relative risk" axis label is accurate. In the submitted
   figure the ATE/ATT/ATC panels were produced from risk differences, which is
   why the effect estimates quoted in the text were exponentiated risk
   differences (see the correction in the Results).

     ATE   RR = E[Y(1)]      / E[Y(0)]
     ATT   RR = E[Y(1)|A=1]  / E[Y(0)|A=1]
     ATC   RR = E[Y(1)|A=0]  / E[Y(0)|A=0]

2. THE ITE PANEL. An individual treatment effect does not have a single point
   estimate and standard error per negative control, so it cannot be drawn as a
   funnel plot; the submitted ITE column showed a degenerate spike at RR = 1
   with EASE = 0.00 because the underlying quantity was the rounded predicted
   probability of the control outcome, which is identically zero for a control
   with prevalence below one half. The column is replaced by the distribution of
   estimated individual effects across patients, per negative control. The true
   individual effect is zero for every patient, so the correct picture is a
   spike at zero and any visible width is spurious heterogeneity.

Shading follows the OHDSI empirical calibration convention: the dashed lines are
the conventional significance boundary |logRR| = 1.96 SE, and the shaded region
is the acceptance region of the fitted empirical null,
|logRR - mu| <= 1.96 sqrt(sigma^2 + SE^2).

    python make_funnel.py
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

from rwet.causal_core import (PS_CLIP, _standardise, crossfit_nuisance,  # noqa: E402
                         ease_from_null, fit_empirical_null)
from rwet.ite_core import dr_learner  # noqa: E402
from rwet.nco_split import three_way  # noqa: E402

AUX = str(paths.MIMIC)
WORK = str(paths.WORK)
OUT = str(paths.RESULTS)
FIGDIR = str(paths.FIGURES)

LABS = ["Lactate", "pCO2", "pH", "pO2", "Bicarbonate", "Calcium_Total", "Chloride",
        "Creatinine", "Magnesium", "Potassium", "Sodium", "Urea_Nitrogen",
        "Hemoglobin", "Platelet_Count", "White_Blood_Cells"]
CATS = ["gender", "race", "first_careunit"]

ROWS = [("Unadjusted", "Unadjusted"),
        ("X (covariates)", "Covariate\nadjustment"),
        ("Repr [E0]", "Ours\nw/o debias"),
        ("Repr [E]", "Ours")]

# Generated width. The figure is included at width=\textwidth (6.14in), so this
# is also the factor every font size is inflated by -- see plotstyle. It was
# 15.0in, a 0.41 shrink that printed 9pt labels at 3.7pt, which is Reviewer 1's
# readability complaint.
FIG_W = 13.0
ps.use(FIG_W)     # shared style, identical to the Penn Medicine figure


# ---------------------------------------------------------------------------
# log risk ratios for each target population
# ---------------------------------------------------------------------------
def logrr_all(F, A, Y, n_splits=5, seed=0):
    """logRR + influence-function SE for the ATE, ATT and ATC populations."""
    n = len(A)
    e, mu1, mu0 = crossfit_nuisance(F, A, Y, n_splits, seed)
    out = {}

    psi1 = mu1 + A * (Y - mu1) / e
    psi0 = mu0 + (1 - A) * (Y - mu0) / (1 - e)
    m1, m0 = max(psi1.mean(), 1e-10), max(psi0.mean(), 1e-10)
    ic = (psi1 - m1) / m1 - (psi0 - m0) / m0
    out["ATE"] = (np.log(m1 / m0), ic.std(ddof=1) / np.sqrt(n))

    pA = max(A.mean(), 1e-10)
    a1 = A * Y / pA                                             # E[Y(1)|A=1]
    a0 = (A * mu0 + (1 - A) * e / (1 - e) * (Y - mu0)) / pA     # E[Y(0)|A=1]
    m1, m0 = max(a1.mean(), 1e-10), max(a0.mean(), 1e-10)
    ic = (a1 - m1) / m1 - (a0 - m0) / m0
    out["ATT"] = (np.log(m1 / m0), ic.std(ddof=1) / np.sqrt(n))

    pC = max(1 - A.mean(), 1e-10)
    c1 = ((1 - A) * mu1 + A * (1 - e) / e * (Y - mu1)) / pC     # E[Y(1)|A=0]
    c0 = (1 - A) * Y / pC                                       # E[Y(0)|A=0]
    m1, m0 = max(c1.mean(), 1e-10), max(c0.mean(), 1e-10)
    ic = (c1 - m1) / m1 - (c0 - m0) / m0
    out["ATC"] = (np.log(m1 / m0), ic.std(ddof=1) / np.sqrt(n))

    return out


# ---------------------------------------------------------------------------
def funnel_panel(ax, est, se, mu, sigma, ease, title=None, ylab=False):
    """Thin wrapper so this figure and the Penn Medicine one share a routine."""
    return ps.funnel(ax, est, se, mu=mu, sigma=sigma, ease=ease,
                     title=title, ylab=ylab)


# Four, not five. Each control needs a full text line for its name, the panel is
# one row of a six-row grid, and the annotation box takes about a third of the
# panel height; at five the names printed closer together than their own line
# height and the descenders touched. The annotated spread is the mean over the
# whole admitted panel regardless of how many bars are drawn, so this is a
# choice about how many exemplars to show and changes no reported quantity.
TOP_N_CONTROLS = 4


def _short(c, n=18):
    """Trim a control name for the axis, marking the trim with an ellipsis so it
    reads as deliberate rather than as a rendering fault."""
    s = str(c).replace("_", " ")
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def ite_panel(ax, per, title=None, top_n=TOP_N_CONTROLS):
    """Spread of estimated individual effects, for the largest negative controls.

    A valid negative control cannot be caused by the treatment, so its individual
    effect is zero for EVERY patient, not merely on average. The bar therefore
    ought to collapse to the line at zero, and its width is heterogeneity the
    pipeline has invented. Only the controls with the most events are drawn: the
    rest carry too few events for the spread to mean anything, and plotting all
    of them obscures the ones that do.
    """
    k = ps.scale()
    ax.set_facecolor(ps.PANEL_BG)
    lim = 0.16
    shown = per.sort_values("n_events", ascending=False).head(top_n)
    shown = shown.sort_values("n_events")
    ys = np.arange(len(shown))
    ax.axvline(0.0, color="black", lw=1.6 * k, zorder=5)
    for y, (_, r) in zip(ys, shown.iterrows()):
        ax.plot([r["q10"], r["q90"]], [y, y], color=ps.POINT, lw=3.0 * k,
                alpha=0.6, solid_capstyle="round", zorder=4)
        ax.plot([r["mean"]], [y], "o", ms=4.2 * k, color=ps.POINT_EDGE, zorder=6)
    labels = [_short(c) for c in shown["cluster"]]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=ps.pt(ps.TICK - 0.5))
    # labels on the right, so they do not intrude on the ATC panel to the left
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-lim, lim)
    # Only enough headroom for the annotation box. The control names are one
    # text line each and the panel is short, so any slack above the topmost bar
    # is taken directly out of the spacing between those names: at the previous
    # (-0.8, n+1.5) the five labels were closer together than their own line
    # height and the descenders touched the line below.
    ax.set_ylim(-0.6, len(shown) + 1.8)
    ax.set_xticks([-0.15, 0.0, 0.15])
    ax.set_xlabel("Individual effect", labelpad=2 * k)
    if title:
        ax.set_title(title, pad=6 * k)
    het = per["sd"].mean()
    ax.text(0.035, 0.975, f"spread {het:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=ps.pt(ps.ANNOT), zorder=20,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#888888",
                      lw=0.6 * k))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    obj_idx, val_idx, test_idx, clusters, _ = three_way()
    eval_idx = np.sort(np.concatenate([val_idx, test_idx]))

    z = np.load(os.path.join(WORK, "cohort_nco.npz"), allow_pickle=True)
    nco = pd.DataFrame(z["W"], index=z["ids"], columns=list(z["clusters"]))
    co = pd.read_csv(os.path.join(AUX, "downstream_MIMIC_IV_data.csv"),
                     index_col=0, low_memory=False)
    co = co.drop_duplicates(subset="subject_id", keep="first")
    co = co.set_index(co["subject_id"].astype(np.int64))

    def load(tag):
        tr = pd.read_csv(os.path.join(WORK, f"treat_features_{tag}.csv"), header=None)
        ct = pd.read_csv(os.path.join(WORK, f"control_features_{tag}.csv"), header=None)
        f = pd.concat([tr, ct], ignore_index=True).drop_duplicates(subset=0, keep="first")
        return f.set_index(f[0].astype(np.int64)).drop(columns=[0])

    reps = {t: load(t) for t in ("E", "E0")}
    common = co.index.intersection(nco.index)
    for v in reps.values():
        common = common.intersection(v.index)
    co, nco = co.loc[common], nco.loc[common]
    A = co["dex_flag"].to_numpy(float)

    lab = co[LABS].apply(pd.to_numeric, errors="coerce")
    miss = lab.isna().astype(float); miss.columns = [f"{c}_m" for c in lab.columns]
    lab = lab.fillna(lab.median())
    age = pd.to_numeric(co["anchor_age"], errors="coerce"); age = age.fillna(age.median())
    dm = pd.get_dummies(co[CATS].astype(str), drop_first=True).astype(float)
    X = pd.concat([lab, miss, age.rename("age"), dm], axis=1).to_numpy(float)

    sets = {"Unadjusted": np.zeros((len(A), 1)), "X (covariates)": X,
            "Repr [E0]": reps["E0"].loc[common].to_numpy(float),
            "Repr [E]": reps["E"].loc[common].to_numpy(float)}

    Wm = nco.to_numpy(float)
    names = [clusters[j] for j in eval_idx]

    # ---- per-NCO estimates on the log-RR scale, per estimand ----------------
    cache = os.path.join(OUT, "funnel_nco_estimates.csv")
    if os.path.exists(cache) and "--recompute" not in sys.argv:
        nc = pd.read_csv(cache)
        print(f"  reusing {cache} (pass --recompute to refit)", flush=True)
    else:
        recs = []
        for mname, F in sets.items():
            for jj, j in enumerate(eval_idx):
                w = Wm[:, j].astype(float)
                if w.sum() < 25:
                    continue
                try:
                    r = logrr_all(F, A, w)
                except Exception:
                    continue
                for est_name, (e_, s_) in r.items():
                    if np.isfinite(e_) and np.isfinite(s_) and s_ > 0:
                        recs.append(dict(method=mname, estimand=est_name,
                                         cluster=names[jj], logrr=e_, se=s_))
            print(f"  {mname}: done", flush=True)
        nc = pd.DataFrame(recs)
        nc.to_csv(cache, index=False)

    # ---- ITE per-NCO distributions -----------------------------------------
    ite_path = os.path.join(OUT, "ite_mimic_per_nco.csv")
    ite = pd.read_csv(ite_path) if os.path.exists(ite_path) else pd.DataFrame()

    # The unadjusted analysis is not in stage6 output; add it here. With no
    # covariates the fitted individual effect is necessarily constant, which is
    # the correct answer under the null and makes the contrast with the
    # adjusted rows interpretable.
    missing = [k for k, _ in ROWS if ite.empty or not (ite["method"] == k).any()]
    add = []
    for k in missing:
        F = sets[k]
        for jj, j in enumerate(eval_idx):
            w = Wm[:, j].astype(float)
            if w.sum() < 25:
                continue
            try:
                t = dr_learner(F, A, w, seed=0)
            except Exception:
                continue
            if np.all(np.isfinite(t)):
                add.append(dict(method=k, cluster=names[jj], n_events=float(w.sum()),
                                mean_abs=float(np.mean(np.abs(t))),
                                sd=float(np.std(t, ddof=1)), mean=float(np.mean(t)),
                                q10=float(np.quantile(t, 0.10)),
                                q90=float(np.quantile(t, 0.90))))
        print(f"  ITE for {k}: done", flush=True)
    if add:
        ite = pd.concat([ite, pd.DataFrame(add)], ignore_index=True)
        ite.to_csv(ite_path, index=False)

    # ---- figure -------------------------------------------------------------
    cols = ["ATE", "ATT", "ATC", "ITE"]
    fig, axes = plt.subplots(len(ROWS), 4,
                             figsize=(FIG_W, 2.55 * len(ROWS)))
    summary = []
    for i, (key, label) in enumerate(ROWS):
        for jcol, cname in enumerate(cols):
            ax = axes[i, jcol]
            title = cname if i == 0 else None
            if cname == "ITE":
                if not ite.empty and (ite["method"] == key).any():
                    ite_panel(ax, ite[ite["method"] == key], title=title)
                else:
                    ax.axis("off")
                    if title:
                        ax.set_title(title, pad=6 * ps.scale())
                continue
            sub = nc[(nc["method"] == key) & (nc["estimand"] == cname)]
            mu, sg, k = fit_empirical_null(sub["logrr"].to_numpy(),
                                           sub["se"].to_numpy())
            ea = ease_from_null(mu, sg)
            funnel_panel(ax, sub["logrr"].to_numpy(), sub["se"].to_numpy(),
                         mu, sg, ea, title=title, ylab=(jcol == 0))
            summary.append(dict(method=key, estimand=cname, mu=mu, sigma=sg,
                                ease=ea, k=k))
        ps.row_label(axes[i, 0], label, x=-0.86)

    fig.subplots_adjust(left=0.152, right=0.995, top=0.930, bottom=0.070,
                        hspace=0.60, wspace=0.42)
    dest = os.path.join(FIGDIR, "mimicease_corrected.png")
    fig.savefig(dest, bbox_inches="tight")
    fig.savefig(dest.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "funnel_ease_summary.csv"), index=False)
    print(f"\n{sm.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
