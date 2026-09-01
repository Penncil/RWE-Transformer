"""Method x estimand funnel grid for the MIMIC-IV re-derived cohort.

Drawn with the shared routine, ps.funnel(), so this figure, the Penn Medicine
funnel (make_upenn_funnel.py) and the superseded MIMIC figure are the same plot
in the same style: log risk-ratio on x, standard error on y, the shaded region
the acceptance region of the fitted empirical null and the dashed lines the
conventional |logRR| = 1.96 SE boundary.

Numbers come from the same evaluation as the two MIMIC tables
(results/full_insample_ease_<ARM>.csv and full_insample_nco_<ARM>.csv), so the
EASE annotated in a panel is the EASE tabulated for that row.

    python make_funnel_grid.py

Output: figures/funnel_grid_<ARM>.{pdf,png}
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

ARM = os.environ.get("ARM", "F64").upper()
OUT = str(paths.RESULTS)
FIGS = str(paths.FIGURES)
FIGDIR = str(paths.FIGURES)

# generated wide and included at \textwidth; ps.use() inflates type by the
# reduction factor so it prints at the sizes declared in plotstyle
FIG_W = 13.0
ROW_H = 2.20

# Unadjusted is first so the rows match Table~\ref{tab:mimic-rederived-ease}
# method for method; it has no individual-effect arm, and that panel is left
# blank rather than filled with a quantity the analysis cannot produce.
ROWS = [("Unadjusted", "Unadjusted"),
        ("X", "Baseline"),
        ("BCAUSS", "BCAUSS"),
        ("CausalEGM", "CausalEGM"),
        ("Z0", "Ours\nw/o debias"),
        ("XZ", "Ours")]
COLS = ["ATT", "ATE", "ATC", "ITE"]
HEADLINE = "XZ"


def main():
    ps.use(FIG_W)
    os.makedirs(FIGS, exist_ok=True)

    nc = pd.read_csv(os.path.join(OUT, f"full_insample_nco_{ARM}.csv"))
    nc = nc[np.isfinite(nc.logrr) & np.isfinite(nc.se) & (nc.se > 0) & (nc.se < 5)]
    tab = pd.read_csv(os.path.join(OUT, f"full_insample_ease_{ARM}.csv"))
    tab = tab[tab["seed"] > 0]

    # points come from one seed; take the one whose headline EASE is closest to
    # the multi-seed mean rather than the first, which on this arm is the worst
    h = tab[(tab["method"] == HEADLINE) & (tab["estimand"] == "ATE")]
    seed = int(h.loc[(h["ease"] - h["ease"].mean()).abs().idxmin(), "seed"])
    nc = nc[nc["seed"] == seed]
    print(f"representative seed {seed}")

    ite = pd.read_csv(os.path.join(OUT, f"ite_grid_{ARM}.csv"))

    drawn = [k for k, _ in ROWS]
    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(FIG_W, ROW_H * len(ROWS)))
    summary = []
    for j, est in enumerate(COLS):
        if est == "ITE":
            # not a funnel: an individual effect has no single point estimate
            # and standard error per control, so this column shows the spread
            # of the estimated individual effects, whose correct value is zero
            # for every patient
            for i, (key, _lab) in enumerate(ROWS):
                ax = axes[i, j]
                sub = ite[ite["method"] == key]
                if not len(sub):
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(est, pad=6 * ps.scale())
                    continue
                ite_panel(ax, sub, title=(est if i == 0 else None))
                summary.append(dict(method=key, estimand="ITE",
                                    mu=float(sub["mean"].mean()),
                                    sigma=float(sub["sd"].mean()),
                                    ease_seed=np.nan,
                                    ease_mean=float(sub["sd"].mean()),
                                    k=len(sub)))
            continue
        d_est = nc[(nc["estimand"] == est) & (nc["method"].isin(drawn))]
        keep = d_est.groupby("cluster")["method"].nunique()
        keep = set(keep[keep == d_est["method"].nunique()].index)
        d_est = d_est[d_est["cluster"].isin(keep)]
        print(f"  {est}: {len(keep)} controls")

        for i, (key, lab) in enumerate(ROWS):
            ax = axes[i, j]
            sub = d_est[d_est["method"] == key]
            mu, sg, k = fit_empirical_null(sub.logrr.to_numpy(),
                                           sub.se.to_numpy())
            # annotate the multi-seed mean, not this seed's own fit
            g = tab[(tab["method"] == key) & (tab["estimand"] == est)]["ease"]
            ease = g.mean() if len(g) else ease_from_null(mu, sg)
            ps.funnel(ax, sub.logrr.to_numpy(), sub.se.to_numpy(),
                      mu=mu, sigma=sg, ease=ease,
                      title=(est if i == 0 else None), ylab=(j == 0))
            summary.append(dict(method=key, estimand=est, mu=mu, sigma=sg,
                                ease_seed=ease_from_null(mu, sg),
                                ease_mean=ease, k=k))
    for i, (_key, lab) in enumerate(ROWS):
        # clear of the "Standard error" axis label, which the two-line method
        # names were running into
        ps.row_label(axes[i, 0], lab, x=-1.05)

    # the ITE column carries its control names on the right, so the right
    # margin has to leave room for them
    fig.subplots_adjust(left=0.150, right=0.90, top=0.935, bottom=0.062,
                        hspace=0.60, wspace=0.42)
    pd.DataFrame(summary).to_csv(
        os.path.join(OUT, f"funnel_grid_{ARM}_fits.csv"), index=False)
    for d in (FIGS, FIGDIR):
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, "funnel_grid.png" if d == FIGDIR
                            else f"funnel_grid_{ARM}.png")
        fig.savefig(base, bbox_inches="tight")
        fig.savefig(base.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"-> {base.replace('.png', '.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
