"""Emit the ADRD EASE table (Table 3) from the Figure 5 fits.

One drug, one control panel, the same panel for every method, so the table and
Figure~\\ref{fig:adrd-ease} cannot drift apart.

    DRUG=40790 python make_adrd_table.py
"""
from __future__ import annotations

import os

from rwet import paths

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DRUG = os.environ.get("DRUG", "40790")
OUT = str(paths.RESULTS)
COLS = [("X", r"Baseline"),
        ("BCAUSS", r"Bcauss"),
        ("CausalEGM", r"CausalEGM"),
        ("Z", r"\methodName\ (rep.)"),
        ("XZ", r"\methodName\ (+cov.)")]
ROWS = ["ATT", "ATE", "ATC", "ITE"]


def main():
    f = pd.read_csv(os.path.join(OUT, f"fig5_{DRUG}_fits.csv"))
    ease = f.pivot(index="estimand", columns="method", values="ease")
    kk = f.pivot(index="estimand", columns="method", values="k")

    lines = [r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
             "Outcome & " + " & ".join(c[1] for c in COLS) + r" \\",
             r"\hline"]
    for r in ROWS:
        best = min(ease.loc[r, k] for k, _ in COLS)
        cells = []
        for k, _ in COLS:
            v = ease.loc[r, k]
            s = f"{v:.3f}"
            cells.append(rf"\textbf{{{s}}}" if v == best else s)
        k_r = int(kk.loc[r, COLS[0][0]])
        lines.append(f"{r} ($k$={k_r}) & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    body = "\n".join(lines)
    p = os.path.join(OUT, f"adrd_table_{DRUG}.tex")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    print(body)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
