"""No-hold-out arm at the submitted network size: hidden 64, 3 epochs, 4 seeds.

The hidden-128 arm and the legacy `model_debias` differ in two things at once
(width and training length), so neither isolates the effect of width. This runs
hidden 64 at a matched schedule across four seeds, with the lambda=0 companion
trained at the SAME width -- otherwise the "ours w/o debias" row would be a
different architecture from the row it is meant to ablate.

8 trainings + 8 extractions, two at a time.

    python run_h64e3.py
"""
from __future__ import annotations

import os

from rwet import paths
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(paths.WORK)
SEEDS = [201, 202, 203, 204]
CONCURRENT = int(os.environ.get("CONCURRENT", "2"))
HIDDEN, LAYERS, LR, EPOCHS, BATCHES = 64, 2, "3e-3", 3, 30000


def tag(seed, lam):
    return f"C{seed}L{int(lam)}F64"


def trained(t):
    return os.path.exists(os.path.join(WORK, f"model_{t}", "config.json"))


def extracted(t):
    return all(os.path.exists(os.path.join(WORK, f"anchor_h48_{arm}_{t}.csv"))
               for arm in ("treat", "control"))


def pool(jobs, label):
    todo, running, t0 = list(jobs), [], time.time()
    while todo or running:
        while todo and len(running) < CONCURRENT:
            t, cmd, log = todo.pop(0)
            lf = open(os.path.join(HERE, log), "w")
            running.append((t, subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)))
            print(f"  [{label}] launched {t}", flush=True)
        time.sleep(20)
        still = []
        for t, p in running:
            if p.poll() is None:
                still.append((t, p))
            else:
                print(f"  [{label}] finished {t} rc={p.returncode} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        running = still


def main():
    tags = [tag(s, lam) for s in SEEDS for lam in (10, 0)]

    train = []
    for s in SEEDS:
        for lam in (10, 0):
            t = tag(s, lam)
            if trained(t):
                continue
            train.append((t, [
                sys.executable, os.path.join(paths.ROOT, "scripts", "train", "stage3b_sweep_train.py"),
                "--tag", t, "--lam", str(lam), "--hidden", str(HIDDEN),
                "--layers", str(LAYERS), "--lr", LR, "--epochs", str(EPOCHS),
                "--max-batches", str(BATCHES), "--seed", str(s),
                "--split", "full"], f"h64_{t}.log"))
    print(f"training {len(train)} / {len(tags)}", flush=True)
    pool(train, "train")

    ext = [(t, [sys.executable, os.path.join(paths.ROOT, "scripts", "train", "test_anchor_effect.py"),
                "--tag", t, "--anchor", "h48"], f"ex_{t}.log")
           for t in tags if not extracted(t)]
    print(f"extracting {len(ext)} / {len(tags)}", flush=True)
    pool(ext, "extract")

    print(f"\ncomplete: {sum(extracted(t) for t in tags)}/{len(tags)}")


if __name__ == "__main__":
    main()
