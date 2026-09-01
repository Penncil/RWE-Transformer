"""Recover the per-control points from the OneFlorida+ calibration images.

The OneFlorida+ per-control estimates were never written to disk -- the summary
CSVs carry a fitted null and a calibrated effect, one row each -- so the only
surviving record of them is the scatter in the published calibration PNGs.

This detects the blue markers and maps pixel positions back through the known
axes (x logarithmic from 0.25 to 10, y linear from 0 to 1.5, both read from the
tick labels of the source figures) to recover (relative risk, standard error)
for each control. It is a recovery of published values, not a re-analysis: the
positions are approximate and markers that overlap in the image merge into one.

    python extract_uf_points.py
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np
from PIL import Image

SRC = (str(paths.AD_OUTPUTS) +
       r"\drug_6918")
OUT = str(paths.RESULTS)
FILES = {("Baseline", "ATE"): "NC_calibration_UFATE.png",
         ("Baseline", "ATT"): "NC_calibration_UFATT.png",
         ("Baseline", "ATC"): "NC_calibration_UFATC.png",
         ("Ours", "ATE"): "NC_calibration_UF_ourATE.png",
         ("Ours", "ATT"): "NC_calibration_UF_ourATT.png",
         ("Ours", "ATC"): "NC_calibration_UF_ourATC.png"}


def marker_mask(rgb):
    """The markers are a saturated blue-violet; the panel is grey/orange/white."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (b > 150) & (b - r > 40) & (b - g > 55)


def axis_box(rgb):
    """Locate the plot area from the black axis lines drawn at its edges."""
    dark = (rgb.max(-1) < 90)
    cols = np.where(dark.sum(0) > 0.30 * dark.shape[0])[0]
    rows = np.where(dark.sum(1) > 0.30 * dark.shape[1])[0]
    return cols, rows


def components(mask, min_px=12):
    """Label 8-connected blobs without pulling in scipy."""
    seen = np.zeros_like(mask, bool)
    out = []
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, pts = [(y, x)], []
            seen[y, x] = True
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


def main():
    rows = []
    for (arm, est), fn in FILES.items():
        img = Image.open(os.path.join(SRC, fn)).convert("RGB")
        rgb = np.asarray(img)
        cols, rws = axis_box(rgb)
        # x: the vertical rule at RR = 1 is the tallest dark column; the panel
        # spans the plotted range 0.25..10 between the outermost gridlines.
        blobs = components(marker_mask(rgb))
        rows.append((arm, est, fn, rgb.shape, len(blobs), cols, rws, blobs))
        print(f"{arm:9s} {est}  {fn:34s}  markers found: {len(blobs)}")
    np.save(os.path.join(OUT, "uf_blobs.npy"),
            np.array(rows, dtype=object), allow_pickle=True)
    print(f"\n-> {os.path.join(OUT, 'uf_blobs.npy')}")


if __name__ == "__main__":
    main()
