"""Per-control individual-effect spreads for the funnel grid's ITE column.

A valid negative control cannot be caused by the treatment, so its individual
effect is zero for every patient, not merely on average. The spread of the
estimated individual effects is therefore heterogeneity the pipeline invented,
and is the quantity the ITE column shows.

Computed on the same models as the two MIMIC tables and the funnel columns:
X, Z0 and XZ come from the DR-learner on the seed's own representation, BCAUSS
and CausalEGM from their predicted counterfactual risks (arm-independent, so
reused from results/ite_baselines_per_nco.csv).

    SEED=203 ARM=F64 python make_ite_grid.py

Output: results/ite_grid_<ARM>.csv
"""
from __future__ import annotations

import os

from rwet import paths
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


ARM = os.environ.get("ARM", "F64").upper()
SEED = os.environ.get("SEED", "203")
os.environ["PANEL"] = "all"
os.environ["ZTAG"] = f"C{SEED}L10F64" if ARM == "F64" else f"C{SEED}L10FULL"
os.environ["Z0TAG"] = f"C{SEED}L0F64" if ARM == "F64" else f"C{SEED}L0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rwet.ite_core import dr_learner  # noqa: E402
from rwet.make_funnel_corrected import build  # noqa: E402

OUT = str(paths.RESULTS)
WORK = str(paths.WORK)
LM_H = 48.0
MIN_EV = 25          # the admitted panel, as elsewhere


def main():
    print(f"ARM={ARM} seed={SEED}  Z={os.environ['ZTAG']}  Z0={os.environ['Z0TAG']}")
    sets, A, Y, _n, _p = build()

    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    full = pd.DataFrame(z["W"], index=z["ids"], columns=list(z["clusters"]))
    coh = pd.read_csv(os.path.join(WORK, "cohort_corrected.csv"), index_col=0)
    los_h = coh["los_d"] * 24
    at_risk = (los_h > LM_H) & ~(coh["t_af_h"].notna() & (coh["t_af_h"] <= LM_H))
    full = full.reindex(coh[at_risk].index).fillna(0)
    full = full.iloc[-len(A):] if len(full) > len(A) else full
    cols = [c for c in full.columns
            if MIN_EV <= full[c].sum() <= len(full) - MIN_EV]
    print(f"{len(cols)} controls\n", flush=True)

    rows = []
    for key in ("X", "Z0", "Z", "XZ"):
        F = sets[key]
        ok = 0
        for c in cols:
            w = full[c].to_numpy(float)
            try:
                t = dr_learner(F, A, w, seed=0)
            except Exception:
                continue
            if not np.all(np.isfinite(t)):
                continue
            ok += 1
            rows.append(dict(method=key, cluster=c, n_events=float(w.sum()),
                             mean_abs=float(np.mean(np.abs(t))),
                             sd=float(np.std(t, ddof=1)),
                             mean=float(np.mean(t)),
                             q10=float(np.quantile(t, 0.10)),
                             q90=float(np.quantile(t, 0.90))))
        print(f"  {key}: {ok}/{len(cols)} controls", flush=True)

    d = pd.DataFrame(rows)
    bl = os.path.join(OUT, "ite_baselines_per_nco.csv")
    if os.path.exists(bl):
        b = pd.read_csv(bl)
        b = b[b["method"].isin(("BCAUSS", "CausalEGM"))
              & b["cluster"].isin(cols)]
        d = pd.concat([d, b], ignore_index=True)
        print(f"  merged {b['method'].nunique()} comparators")

    p = os.path.join(OUT, f"ite_grid_{ARM}.csv")
    d.to_csv(p, index=False)
    print(f"\n-> {p}")
    print(d.groupby("method")["sd"].agg(["count", "mean"]).round(4).to_string())


if __name__ == "__main__":
    main()
