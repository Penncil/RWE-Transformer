"""Does the orientation of the penalty statistic matter on real data?

The released `partial_corr.py` minimises

    torch.mean(torch.corrcoef(torch.hstack([res_W, res_T]))[:, 0] ** 2)

which, because `torch.corrcoef` treats rows as variables, correlates patients
with one another rather than negative-control residuals with treatment
residuals. The corrected form transposes and reads the d_W x d_A block. All
results reported in the paper use the corrected form; this compares the two
directly, which had not been done.

Three models, identical in every respect but the penalty, at one seed:

    corrected    the form used throughout the paper
    asreleased   the form in the public repository
    none         lambda = 0, the control both are measured against

Configuration matches the reported F64 arm exactly: hidden 64, 2 layers,
lr 3e-3, 3 epochs, 30,000 batches, no hold-out split. Runs are sequential; two
training jobs on one GPU contend and have failed before.

    python run_penalty_comparison.py
"""
from __future__ import annotations

import os

from rwet import paths
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(paths.WORK)
SEED = 777
ARMS = [("VCOR", "corrected", "10"),
        ("VREL", "asreleased", "10"),
        ("VNON", "corrected", "0")]     # lam=0: the statistic is never applied


def trained(tag):
    return os.path.exists(os.path.join(WORK, f"model_{tag}", "config.json"))


def extracted(tag):
    return all(os.path.exists(os.path.join(WORK, f"anchor_h48_{a}_{tag}.csv"))
               for a in ("treat", "control"))


def run(cmd, log):
    with open(os.path.join(HERE, log), "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    return p.returncode


def main():
    t0 = time.time()
    for tag, pen, lam in ARMS:
        if trained(tag):
            print(f"[{tag}] already trained, skipping", flush=True)
            continue
        print(f"[{tag}] training  penalty={pen} lam={lam} "
              f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
        rc = run([sys.executable, os.path.join(paths.ROOT, "scripts", "train", "stage3b_sweep_train.py"),
                  "--tag", tag, "--penalty", pen, "--lam", lam,
                  "--hidden", "64", "--layers", "2", "--lr", "3e-3",
                  "--epochs", "3", "--max-batches", "30000",
                  "--seed", str(SEED), "--split", "full"], f"pen_{tag}.log")
        print(f"[{tag}] train rc={rc}", flush=True)

    for tag, _pen, _lam in ARMS:
        if extracted(tag):
            print(f"[{tag}] already extracted, skipping", flush=True)
            continue
        print(f"[{tag}] extracting ({(time.time()-t0)/60:.0f} min elapsed)",
              flush=True)
        rc = run([sys.executable, os.path.join(paths.ROOT, "scripts", "train", "test_anchor_effect.py"),
                  "--tag", tag, "--anchor", "h48"], f"pen_ex_{tag}.log")
        print(f"[{tag}] extract rc={rc}", flush=True)

    done = [t for t, _, _ in ARMS if extracted(t)]
    print(f"\ncomplete: {len(done)}/{len(ARMS)} -> {done}"
          f"  ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
