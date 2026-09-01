"""Split with a LARGER objective panel: 12 controls train the penalty, 7 held out.

The corrected split allocates 7 / 6 / 6 by dealing round-robin over prevalence,
so the penalty trains on 7 of 19 usable controls -- against 40 of 40 in the
submitted pipeline, which held nothing out. It is also badly concentrated:
sleep_apnea alone carries 1,863 of the objective panel's 2,965 events, so the
penalty is largely learning to decorrelate treatment from one condition.

This deals the same prevalence-ordered list into 12 objective and 7 evaluation
instead, keeping the two strictly disjoint. The penalty gets roughly twice the
signal and EASE stays non-circular; the price is that the evaluation null is
fitted on 7 controls rather than 12, so its EASE values are NOT comparable with
the tables built on the corrected split.

Fixed before any result was computed: allocation is deterministic given the
prevalence ordering and SEED below, and is not to be re-drawn.

  objective  (12) -> enters the training penalty
  evaluation  (7) -> never enters the penalty; the only panel behind a reported
                     EASE for this arm

The trainer expects three panels. Evaluation is returned as both the validation
and test panel: it is only logged during training, never optimised against, so
this does not make it circular -- but the reported EASE for this arm must be
read as coming from 7 controls.
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np

WORK = str(paths.WORK)
SEED = 20260814          # same as nco_split_corrected, deliberately
MIN_EVENTS = 25
N_OBJECTIVE = 12


def load_panel():
    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    return z["ids"], z["W"], list(z["clusters"])


def three_way():
    ids, W, clusters = load_panel()
    n = len(W)
    usable = np.array([j for j in range(W.shape[1])
                       if MIN_EVENTS <= W[:, j].sum() <= n - MIN_EVENTS])
    usable = usable[np.argsort(-W[:, usable].sum(0))]

    # Spread the evaluation slots evenly THROUGH the prevalence ordering, so
    # both panels span large and small controls. A naive "first 12" would put
    # every high-event control in the objective panel and leave the evaluation
    # null to be fitted on the seven rarest, which cannot support a fit.
    m, n_ev = len(usable), len(usable) - N_OBJECTIVE
    obj, ev = [], []
    for k, j in enumerate(usable):
        if int((k + 1) * n_ev / m) > int(k * n_ev / m):
            ev.append(j)
        else:
            obj.append(j)

    rng = np.random.default_rng(SEED)
    obj = list(obj)
    ev = list(ev)
    rng.shuffle(obj)
    rng.shuffle(ev)
    return np.sort(obj), np.sort(ev), np.sort(ev), clusters, W


if __name__ == "__main__":
    obj, val, test, clusters, W = three_way()
    ev = W.sum(0).astype(int)
    print(f"objective  ({len(obj)}): {ev[list(obj)].sum():,} events")
    for j in sorted(obj, key=lambda j: -ev[j]):
        print(f"    {clusters[j]:26s} {ev[j]:6d}")
    print(f"evaluation ({len(val)}): {ev[list(val)].sum():,} events")
    for j in sorted(val, key=lambda j: -ev[j]):
        print(f"    {clusters[j]:26s} {ev[j]:6d}")
    print(f"\ndisjoint: {not (set(obj) & set(val))}")
