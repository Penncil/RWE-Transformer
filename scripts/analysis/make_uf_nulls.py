"""Lift the OneFlorida+ fitted nulls out of the restricted summary files.

Figure 8 needs one number per panel: the mean of the fitted empirical null. Of
the OneFlorida+ analysis only the calibrated summaries survive --
`results_calibrated_UF<EST>.csv` for the baseline arm and
`results_calibrated_UF_our<EST>.csv` for ours -- each a single row carrying that
mean in its `bias_est` column. The per-control estimates were never written to
disk, so nothing here can be refitted; this script exists so the six values the
figure draws have a stated origin in a file rather than sitting in the
repository as literals nobody can trace.

The fitted dispersion is taken as zero. It does not survive in these files, but
in every published panel the reported expected absolute systematic error equals
|bias_est| to the two decimals printed, and EASE exceeds |mu| strictly whenever
sigma > 0, so sigma was zero or below the rounding of the printed value. Taking
it as zero therefore reproduces the published quantity and, where it errs, errs
by understating the systematic error of BOTH arms equally.

`results_calibrated_UF_b1<EST>.csv` is not read: it duplicates the `UF_our`
values rather than providing a third arm.

    python make_uf_nulls.py

Output: results/uf_null_fits.csv
"""
from __future__ import annotations

import os

from rwet import paths

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(paths.RESULTS)
SRC = (str(paths.AD_OUTPUTS) +
       r"\drug_6918")
ARMS = [("Baseline", "UF"), ("Ours", "UF_our")]
ESTIMANDS = ["ATT", "ATE", "ATC"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, missing = [], []
    for arm, prefix in ARMS:
        for est in ESTIMANDS:
            p = os.path.join(SRC, f"results_calibrated_{prefix}{est}.csv")
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
                continue
            d = pd.read_csv(p)
            mu = float(d["bias_est"].dropna().iloc[0])
            rows.append(dict(arm=arm, estimand=est, mu=mu, sigma=0.0,
                             ease=abs(mu), source=os.path.basename(p)))
    if missing:
        raise SystemExit("missing restricted summaries: " + ", ".join(missing))

    t = pd.DataFrame(rows).sort_values(["arm", "estimand"])
    p = os.path.join(OUT, "uf_null_fits.csv")
    t.to_csv(p, index=False)
    print(t.to_string(index=False))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
