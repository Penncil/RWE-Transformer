"""Figure 5 for the resubmission: ADRD, drug 40790, five methods, four tasks.

Structure matches the submitted figure -- baseline, the two neural comparators,
the representation, and the representation with covariates -- so the revision is
read against the original row for row.

Estimation. The three population columns come from a cross-fitted AIPW on the
log risk-ratio scale, which weights every patient and discards none; the matched
route the submission used retains only 94 to 163 patients for the concatenated
adjustment set against 228 to 403 for the covariates, and a test holding the
patients fixed showed its control-population advantage came from that selection
rather than from the adjustment. The two neural comparators are estimated from
their predicted counterfactual risks, as in the submission, at a training length
chosen out of sample rather than at their defaults, at which BCAUSS saturates.

The ITE column is the spread of estimated individual effects about their correct
value of zero, from the DR-learner, on the same models.

    DRUG=40790 python make_fig5.py
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
from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402
from rwet.make_funnel import ite_panel  # noqa: E402

DRUG = os.environ.get("DRUG", "40790")
OUT = str(paths.RESULTS)
FIGS = str(paths.FIGURES)
FIGDIR = str(paths.FIGURES)
ROWS = [("X", "Baseline"), ("BCAUSS", "BCAUSS"), ("CausalEGM", "CausalEGM"),
        ("Z", "Ours\n(representation)"), ("XZ", "Ours\n(+ covariates)")]
COLS = ["ATT", "ATE", "ATC", "ITE"]
FIG_W, ROW_H = 13.0, 2.20


def main():
    ps.use(FIG_W)
    os.makedirs(FIGS, exist_ok=True)
    nc = pd.read_csv(os.path.join(OUT, f"adrd_nco_{DRUG}.csv"))
    nc = nc[nc["method"].isin([k for k, _ in ROWS])]
    nc = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]
    keep = (nc.groupby(["estimand", "cluster"])["method"].nunique()
            .rename("n").reset_index())
    keep = keep[keep["n"] == nc["method"].nunique()][["estimand", "cluster"]]
    nc = nc.merge(keep, on=["estimand", "cluster"])
    ite = pd.read_csv(os.path.join(OUT, f"adrd_ite_{DRUG}_fixed.csv"))

    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(FIG_W, ROW_H * len(ROWS)))
    summary = []
    for i, (key, label) in enumerate(ROWS):
        for j, est in enumerate(COLS):
            ax = axes[i, j]
            title = est if i == 0 else None
            if est == "ITE":
                sub = ite[ite["method"] == key]
                if len(sub):
                    ite_panel(ax, sub, title=title)
                    summary.append(dict(method=key, estimand="ITE",
                                        mu=float(sub["mean"].mean()),
                                        sigma=float(sub["sd"].mean()),
                                        ease=float(sub["sd"].mean()),
                                        k=len(sub)))
                else:
                    ax.axis("off")
                continue
            s = nc[(nc["method"] == key) & (nc["estimand"] == est)]
            mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
            ps.funnel(ax, s.logrr.values, s.se.values, mu=mu, sigma=sg,
                      ease=ease_from_null(mu, sg), title=title, ylab=(j == 0))
            ax.text(0.965, 0.955, f"$k$={k}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=ps.pt(ps.ANNOT), color="0.3")
            summary.append(dict(method=key, estimand=est, mu=mu, sigma=sg,
                                ease=ease_from_null(mu, sg), k=k))
        # far enough left to clear the "Standard error" axis label, which the
        # two-line method names were running into
        ps.row_label(axes[i, 0], label, x=-1.15)

    fig.subplots_adjust(left=0.185, right=0.895, top=0.935, bottom=0.062,
                        hspace=0.60, wspace=0.42)
    for d, name in ((FIGS, f"fig5_{DRUG}.png"), (FIGDIR, "adrd_fig5.png")):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"-> {p.replace('.png', '.pdf')}")
    plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, f"fig5_{DRUG}_fits.csv"), index=False)
    lab = dict(ROWS)
    sm["label"] = sm["method"].map(lambda k: lab[k].replace("\n", " "))
    for c in ("ease", "sigma", "k"):
        print(f"\n{c}\n")
        print(sm.pivot(index="label", columns="estimand",
                       values=c).reindex(columns=COLS).round(3).to_string())


if __name__ == "__main__":
    main()
