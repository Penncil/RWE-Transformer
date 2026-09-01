"""Three-way split of the negative-control panel, derived from the REBUILT cohort.

Same construction as nco_split.py -- objective / validation / test kept disjoint,
dealt round-robin by prevalence, fixed seed -- but the usable clusters are those
with enough events in the corrected cohort rather than in the superseded one.

  objective  -> enters the training penalty
  validation -> selects the configuration
  test       -> the only panel behind a reported EASE
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np

WORK = str(paths.WORK)
SEED = 20260814
MIN_EVENTS = 25


def load_panel():
    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    return z["ids"], z["W"], list(z["clusters"])


def three_way():
    ids, W, clusters = load_panel()
    n = len(W)
    usable = np.array([j for j in range(W.shape[1])
                       if MIN_EVENTS <= W[:, j].sum() <= n - MIN_EVENTS])
    usable = usable[np.argsort(-W[:, usable].sum(0))]
    rng = np.random.default_rng(SEED)
    obj, val, test = [], [], []
    for k, j in enumerate(usable):
        [obj, val, test][k % 3].append(j)
    rng.shuffle(obj); rng.shuffle(val); rng.shuffle(test)
    return np.sort(obj), np.sort(val), np.sort(test), clusters, W


if __name__ == "__main__":
    obj, val, test, clusters, W = three_way()
    print(f"usable clusters: {len(obj) + len(val) + len(test)}")
    for name, idx in [("objective", obj), ("validation", val), ("test", test)]:
        names = [clusters[j] for j in idx]
        ev = [int(W[:, j].sum()) for j in idx]
        print(f"{name:11s} ({len(idx)}): "
              + ", ".join(f"{n}({e:,})" for n, e in zip(names, ev)))
