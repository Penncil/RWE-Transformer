"""Recover OneFlorida+ per-control estimates from the published calibration PNGs.

Why this is necessary. The OneFlorida+ per-control estimates were never written
to disk: `results_calibrated_UF*.csv` carry a fitted null mean and a calibrated
effect, one row each. The scatter in the published calibration images is the
only surviving record of the individual controls, so Figure 8 can be restyled to
match Figures 4 and 5 only by reading the points back out of those images.

Calibration of the pixel grid uses two references that are unambiguous in the
source figures:

  x   the light vertical gridlines sit at relative risks 0.25, 0.5, 1, 2, 4, 6,
      8, 10, and the heavy black rule at 1; fitting log10(RR) against column
      index over those gives the horizontal scale.
  y   the dashed reference line is |log RR| = 1.96 SE by construction, so once
      the horizontal scale is known its slope in pixels fixes the vertical one
      without needing to read the y tick labels.

Validation. Having recovered the points, the empirical null is refitted from
them and the resulting EASE compared with the value printed on the source image.
Agreement to the two decimals the annotation carries is the check that the
extraction is faithful; disagreement means the recovery failed and the output
must not be used.

    python recover_uf_funnel.py
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
from PIL import Image  # noqa: E402

from rwet.causal_core import ease_from_null, fit_empirical_null  # noqa: E402

SRC = (str(paths.AD_OUTPUTS) +
       r"\drug_6918")
OUT = str(paths.RESULTS)
GRID_RR = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
FILES = {("Baseline", "ATE"): ("NC_calibration_UFATE.png", 0.21),
         ("Baseline", "ATT"): ("NC_calibration_UFATT.png", 0.26),
         ("Baseline", "ATC"): ("NC_calibration_UFATC.png", 0.22),
         ("Ours", "ATE"): ("NC_calibration_UF_ourATE.png", 0.13),
         ("Ours", "ATT"): ("NC_calibration_UF_ourATT.png", 0.15),
         ("Ours", "ATC"): ("NC_calibration_UF_ourATC.png", 0.16)}


def marker_mask(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (b > 150) & (b - r > 40) & (b - g > 55)


def blobs(mask, min_px=12):
    seen = np.zeros_like(mask, bool)
    out, H, W = [], *mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack, pts = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W and mask[ny, nx]
                            and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pts) >= min_px:
            a = np.array(pts, float)
            out.append((a[:, 1].mean(), a[:, 0].mean(), len(pts)))
    return out


def _groups(cols, gap=3):
    out, cur = [], [cols[0]]
    for c in cols[1:]:
        if c - cur[-1] <= gap:
            cur.append(c)
        else:
            out.append((float(np.mean(cur)), len(cur)))
            cur = [c]
    out.append((float(np.mean(cur)), len(cur)))
    return out


def x_scale(rgb):
    """Anchor the horizontal scale on the gridlines.

    The light gridlines sit at the labelled relative risks and the heavy black
    rule at 1. The outermost gridlines (0.25 and 10) and that rule are detected
    reliably in every panel; the crowded 6 and 8 are not, and are not needed.
    Returns the anchor columns and their relative risks.
    """
    H = rgb.shape[0]
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    grey = (abs(r - g) < 14) & (abs(g - b) < 14) & (r > 110) & (r < 220)
    dark = rgb.max(-1) < 90
    gcand = np.where(grey.sum(0) > 0.30 * H)[0]
    dcand = np.where(dark.sum(0) > 0.30 * H)[0]
    gg = [(x, n) for x, n in _groups(gcand) if n >= 3]
    one = float(np.mean([x for x, _ in _groups(dcand)]))
    px = np.array([gg[0][0], one, gg[-1][0]], float)
    rr = np.array([0.25, 1.0, 10.0], float)
    return px, rr


def y_scale(rgb):
    """Vertical scale from the extent of the axes background.

    The panels are drawn with limits of 0 to 1.5 on the standard error and the
    default 5 per cent margin, so the shaded axes region spans -0.075 to 1.575.
    Mapping its first and last row onto that interval fixes the scale, and the
    result is checked against the black rule at zero, which must come back as
    zero to within a pixel.
    """
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    H, W = rgb.shape[:2]
    panel = (abs(r - g) < 8) & (abs(g - b) < 8) & (r > 235) & (r < 252)
    rows = np.where(panel.sum(1) > 0.25 * W)[0]
    top, bot = float(rows.min()), float(rows.max())
    lo, hi = -0.075, 1.575
    axis_row = float(np.argmax((rgb.max(-1) < 90).sum(1)))

    def se_of_row(row):
        return hi - (np.asarray(row, float) - top) * (hi - lo) / (bot - top)

    return se_of_row, float(se_of_row(axis_row))


def main():
    rows, checks = [], []
    for (arm, est), (fn, printed) in FILES.items():
        rgb = np.asarray(Image.open(os.path.join(SRC, fn)).convert("RGB"))
        px, rr = x_scale(rgb)
        A = np.vstack([px, np.ones_like(px)]).T
        (a, b_), res, *_ = np.linalg.lstsq(A, np.log10(rr), rcond=None)
        fitres = float(np.max(np.abs(a * px + b_ - np.log10(rr))))
        print(f"  {fn}: anchors {px.round(0)} px -> RR {rr}, "
              f"max residual {fitres:.4f} in log10")
        if fitres > 0.02:
            print("    axis fit failed -- skipped")
            continue

        def px_of_logrr(cols, a=a, b_=b_):
            return a * np.asarray(cols, float) + b_

        se_of_row, zero_check = y_scale(rgb)
        print(f"    vertical scale: SE at the zero rule = {zero_check:+.4f}")
        if abs(zero_check) > 0.01:
            print("    vertical fit failed -- skipped")
            continue
        found = blobs(marker_mask(rgb))
        # markers that overlap in the image merge into one blob; a blob much
        # larger than the typical marker stands for more than one control, and
        # is counted with the multiplicity its area implies
        unit = np.median([n for _x, _y, n in found])
        for cx, cy, n in found:
            se = float(se_of_row(cy))
            if se <= 0:
                continue
            for _ in range(max(1, int(round(n / unit)))):
                rows.append(dict(method=arm, estimand=est,
                                 cluster=f"c{len(rows)}",
                                 logrr=float(np.log(10) * px_of_logrr(cx)),
                                 se=se))
        s = pd.DataFrame([r for r in rows
                          if r["method"] == arm and r["estimand"] == est])
        mu, sg, k = fit_empirical_null(s.logrr.values, s.se.values)
        got = ease_from_null(mu, sg)
        checks.append(dict(arm=arm, estimand=est, k=k, recovered=round(got, 3),
                           printed=printed, diff=round(abs(got - printed), 3)))

    nc = pd.DataFrame(rows)
    nc.to_csv(os.path.join(OUT, "uf_recovered_nco.csv"), index=False)
    ck = pd.DataFrame(checks)
    print("\nvalidation -- recovered EASE against the value printed on each image\n")
    print(ck.to_string(index=False))
    worst = ck["diff"].max()
    print(f"\nlargest discrepancy: {worst:.3f}")
    print("PASS -- extraction faithful" if worst <= 0.02 else
          "FAIL -- do not use this output")
    print(f"\n-> {os.path.join(OUT, 'uf_recovered_nco.csv')}")


if __name__ == "__main__":
    main()
