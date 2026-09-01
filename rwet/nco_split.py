"""Fixed three-way split of the NCO panel, for the held-out protocol.

  objective  -> enters the training penalty
  validation -> selects the hyperparameter configuration
  test       -> scored, and never seen by either of the above

Keeping these disjoint is what makes hyperparameter selection legitimate on this
arm: the diagnostic is not the quantity that was optimised, nor the quantity the
configuration was chosen on.

This is NOT the split behind the headline tables. Those reproduce the submitted
protocol, which had no hold-out at all: see nco_split_full.py, where objective,
validation and test are the same set, and eval_full_insample.py, whose output is
in-sample by construction and labelled so in the manuscript. This module backs
the held-out cross-check reported alongside it.

The assignment is deterministic. Controls are ordered by prevalence and dealt
round-robin, which balances common and rare controls across the three panels;
there is no random element, and the split is reproducible without a seed.
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np

WORK = str(paths.WORK)
MIN_EVENTS = 25


def load_panel():
    z = np.load(os.path.join(WORK, "cohort_nco.npz"), allow_pickle=True)
    return z["ids"], z["W"], list(z["clusters"])


def three_way():
    """Return (objective, validation, test) index arrays into the cluster axis."""
    ids, W, clusters = load_panel()
    n = len(W)
    usable = np.array([j for j in range(W.shape[1])
                       if MIN_EVENTS <= W[:, j].sum() <= n - MIN_EVENTS])
    # order by prevalence, then deal round-robin so each panel gets a
    # comparable mix of common and rare controls
    usable = usable[np.argsort(-W[:, usable].sum(0))]
    obj, val, test = [], [], []
    for k, j in enumerate(usable):
        [obj, val, test][k % 3].append(j)
    return np.sort(obj), np.sort(val), np.sort(test), clusters, W


if __name__ == "__main__":
    obj, val, test, clusters, W = three_way()
    for name, idx in [("objective", obj), ("validation", val), ("test", test)]:
        names = [clusters[j] for j in idx]
        ev = [int(W[:, j].sum()) for j in idx]
        print(f"{name:11s} ({len(idx)}): "
              + ", ".join(f"{n}({e:,})" for n, e in zip(names, ev)))
