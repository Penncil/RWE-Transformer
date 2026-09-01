"""Figure 8 restyled: the OneFlorida+ zero-shot diagnostic, in the routine used
for Figures 4 and 5, so the three cohorts are drawn alike for the first time.

What is exact and what is recovered. The fitted empirical null survives in
`results_calibrated_UF*.csv` -- one row per panel, carrying its mean -- and in
every panel the published expected absolute systematic error equals the absolute
value of that mean to the two decimals printed, so the fitted dispersion was
zero throughout and the null is fully determined by the surviving files. The
shaded acceptance region, the annotation and the null are therefore the
published quantities, not re-estimates.

The per-control estimates were never written to disk, so the scatter is
recovered from the published calibration images by detecting the markers and
mapping them back through the axes; see `recover_uf_funnel.py`, which validates
the mapping against the printed annotation before writing its output. Recovered
points are approximate, and markers that overlap in the source image cannot be
separated exactly, so the scatter should be read as the published scatter
redrawn rather than as data.

The ITE column of the submitted figure is not reproduced. Its two panels both
annotate 0.01, which is the artefact described in the re-analysis of the
individualised treatment effect: the quantity propagated was the rounded
predicted probability, identically zero for any control with prevalence below a
half, so the column measured nothing.

    python make_uf_figure.py
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

SRC = (str(paths.AD_OUTPUTS) +
       r"\drug_6918")
OUT = str(paths.RESULTS)
FIGS = str(paths.FIGURES)
FIGDIR = str(paths.FIGURES)
ROWS = [("Baseline", "UF", "Baseline"),
        ("Ours", "UF_our", r"\methodName")]
COLS = ["ATE", "ATT", "ATC"]
FIG_W, ROW_H = 10.0, 2.35


def main():
    ps.use(FIG_W)
    os.makedirs(FIGS, exist_ok=True)
    nc = pd.read_csv(os.path.join(OUT, "uf_recovered_nco.csv"))
    # The fitted nulls are a single number per panel, lifted out of the
    # restricted summary files into results/ so that this figure can be redrawn
    # without them; the values are the ones printed in the published panels.
    nulls = pd.read_csv(os.path.join(OUT, "uf_null_fits.csv"))
    mu_of = {(r.arm, r.estimand): float(r.mu) for r in nulls.itertuples()}

    fig, axes = plt.subplots(len(ROWS), len(COLS),
                             figsize=(FIG_W, ROW_H * len(ROWS)))
    summary = []
    for i, (arm, _prefix, label) in enumerate(ROWS):
        for j, est in enumerate(COLS):
            ax = axes[i, j]
            mu = mu_of[(arm, est)]
            s = nc[(nc.method == arm) & (nc.estimand == est)]
            ps.funnel(ax, s.logrr.values, s.se.values, mu=mu, sigma=0.0,
                      ease=abs(mu), title=(est if i == 0 else None),
                      ylab=(j == 0))
            # no k annotation here: overlapping markers in the source image
            # cannot be separated, so a count would be a lower bound presented
            # as though it were the panel size
            summary.append(dict(arm=arm, estimand=est, mu=mu, sigma=0.0,
                                ease=abs(mu), k_recovered=len(s)))
        ps.row_label(axes[i, 0], arm, x=-0.62)

    fig.subplots_adjust(left=0.125, right=0.995, top=0.925, bottom=0.105,
                        hspace=0.58, wspace=0.40)
    for d, name in ((FIGS, "uf_zeroshot.png"), (FIGDIR, "zeroshot_uf_new.png")):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"-> {p.replace('.png', '.pdf')}")
    plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "uf_figure_fits.csv"), index=False)
    print()
    print(sm.pivot(index="arm", columns="estimand",
                   values="ease").reindex(columns=COLS).round(3).to_string())


if __name__ == "__main__":
    main()
