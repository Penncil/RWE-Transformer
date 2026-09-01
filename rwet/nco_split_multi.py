"""12/7 objective/evaluation splits, parameterised by a split seed.

There are C(19,12) = 50,388 ways to choose which 12 controls train the penalty.
The particular one used in nco_split_bigobj.py is arbitrary. This module draws
others so the dependence of the result on that arbitrary choice can be measured.

Every draw keeps both panels spread across the prevalence range, so no split is
degenerate in the way the first attempt was (all common controls in the
objective, leaving a 321-event evaluation panel that cannot fit a null).

    SPLITSEED=1 python -c "import nco_split_multi as m; print(m.three_way()[0])"
"""
from __future__ import annotations

import os

from rwet import paths

import numpy as np

WORK = str(paths.WORK)
MIN_EVENTS = 25
N_OBJECTIVE = 12
SPLITSEED = int(os.environ.get("SPLITSEED", "0"))


def load_panel():
    z = np.load(os.path.join(WORK, "cohort_corrected_nco.npz"), allow_pickle=True)
    return z["ids"], z["W"], list(z["clusters"])


def three_way(splitseed=None):
    ss = SPLITSEED if splitseed is None else splitseed
    ids, W, clusters = load_panel()
    n = len(W)
    usable = np.array([j for j in range(W.shape[1])
                       if MIN_EVENTS <= W[:, j].sum() <= n - MIN_EVENTS])
    usable = usable[np.argsort(-W[:, usable].sum(0))]
    m = len(usable)
    n_ev = m - N_OBJECTIVE

    # Stratify by prevalence: cut the ordered list into n_ev contiguous blocks
    # and take one evaluation control from each, so both panels always span
    # common and rare conditions whatever the seed.
    rng = np.random.default_rng(20260814 + 1000 * ss)
    bounds = np.linspace(0, m, n_ev + 1).astype(int)
    ev = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        if b1 > b0:
            ev.append(int(rng.integers(b0, b1)))
    ev = sorted(set(ev))
    while len(ev) < n_ev:                       # top up if a block collided
        c = int(rng.integers(0, m))
        if c not in ev:
            ev.append(c)
    ev = sorted(ev[:n_ev])
    obj = [k for k in range(m) if k not in set(ev)]

    obj_j = [int(usable[k]) for k in obj]
    ev_j = [int(usable[k]) for k in ev]
    return np.sort(obj_j), np.sort(ev_j), np.sort(ev_j), clusters, W


if __name__ == "__main__":
    obj, val, _t, clusters, W = three_way()
    ev = W.sum(0).astype(int)
    print(f"SPLITSEED={SPLITSEED}")
    print(f"  objective  ({len(obj)}): {ev[list(obj)].sum():,} events")
    print(f"  evaluation ({len(val)}): {ev[list(val)].sum():,} events -> "
          f"{[clusters[j] for j in val]}")
    print(f"  disjoint: {not (set(obj) & set(val))}")
