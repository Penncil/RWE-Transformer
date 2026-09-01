"""No split at all: every admitted control trains the penalty.

This reproduces the submitted protocol on the re-derived cohort. The submission
drew 40 of 53 controls, put all 40 in the penalty, and reported EASE on those
same 40 -- there was no objective/validation/test division. Here the objective
panel is every control that clears the event threshold, and the "validation"
panel is the same set, so the logged partial correlation is an in-sample fit
statistic exactly as the submitted numbers were.

Results from this module are IN-SAMPLE by construction and must be labelled as
such wherever they appear.

    python nco_split_full.py
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np

WORK = str(paths.WORK)
MIN_EVENTS = 25


def load_panel():
    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    return z["ids"], z["W"], list(z["clusters"])


def three_way(splitseed=None):
    _ids, W, clusters = load_panel()
    n = len(W)
    usable = np.array([j for j in range(W.shape[1])
                       if MIN_EVENTS <= W[:, j].sum() <= n - MIN_EVENTS])
    usable = np.sort(usable)
    # objective == validation == test: no hold-out, matching the submission
    return usable, usable, usable, clusters, W


if __name__ == "__main__":
    obj, val, _t, clusters, W = three_way()
    ev = W.sum(0).astype(int)
    print(f"objective = validation = {len(obj)} controls, "
          f"{ev[list(obj)].sum():,} events (NO hold-out)")
    for j in obj:
        print(f"  {clusters[j]:32s} {ev[j]:6,d}")
