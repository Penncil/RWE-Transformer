"""
Sweep-capable pre-training, with an honest selection criterion.

Differences from stage3_train.py:
  * architecture is configurable (--hidden, --layers, --heads)
  * the penalty uses ONLY the objective NCO panel
  * the VALIDATION panel partial correlation is logged every penalty step but
    never enters the loss -- this is the model-selection criterion
  * the next-code (SSL) loss is logged, so a configuration that improves the
    penalty by degrading the representation can be rejected

The test panel is never touched here.

Usage:
    python stage3b_sweep_train.py --tag C --lam 50 --hidden 64 --layers 2
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from rwet import paths
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("FEMR_SRC", ""))  # see README
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import femr.models.config  # noqa: E402
import femr.models.processor  # noqa: E402
import femr.models.tasks  # noqa: E402
import femr.models.tokenizer  # noqa: E402
import femr.models.transformer  # noqa: E402

WORK = str(paths.WORK)
VOCAB = 60000


def partial_corr(res_W, res_T):
    """Mean squared partial correlation, correct matrix orientation."""
    if res_W.shape[1] == 0 or res_W.shape[0] < 4:
        return torch.tensor(0.0, device=res_W.device)
    d_w = res_W.shape[1]
    cm = torch.corrcoef(torch.hstack([res_W, res_T]).T)
    return torch.nan_to_num(cm[:d_w, d_w:], nan=0.0).pow(2).mean()


def partial_corr_asreleased(res_W, res_T):
    """The statistic as it appears in the released `partial_corr.py`.

    Reproduced verbatim so the two can be compared under identical conditions:

        torch.mean(torch.corrcoef(torch.hstack([res_W, res_T]))[:, 0] ** 2)

    Without the transpose, `torch.corrcoef` treats each *patient* as a variable,
    so this is the mean squared correlation between every patient in the batch
    and whichever patient happens to sit in row zero. It is not a function of the
    NCO-treatment relationship at all. Kept here only to run the comparison.
    """
    if res_W.shape[1] == 0 or res_W.shape[0] < 4:
        return torch.tensor(0.0, device=res_W.device)
    cm = torch.corrcoef(torch.hstack([res_W, res_T]))
    return torch.nan_to_num(cm[:, 0], nan=0.0).pow(2).mean()


PENALTY = {"corrected": partial_corr, "asreleased": partial_corr_asreleased}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--penalty", default="corrected",
                    choices=sorted(PENALTY),
                    help="which partial-correlation statistic to minimise")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=25)
    ap.add_argument("--penalty-every", type=int, default=1,
                    help="apply the NCO penalty on one update in every K; "
                         "K=6 gives five plain descent steps then one "
                         "regularisation step")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--split", default="original",
                    choices=["original", "corrected", "bigobj", "multi", "full"],
                    help="which cohort's negative-control panel defines the "
                         "objective/validation split")
    a = ap.parse_args()
    _pc = PENALTY[a.penalty]

    if a.split == "full":
        # no hold-out at all: the submitted protocol
        from rwet.nco_split_full import three_way
    elif a.split == "multi":
        from rwet.nco_split_multi import three_way
    elif a.split == "bigobj":
        from rwet.nco_split_bigobj import three_way
    elif a.split == "corrected":
        from rwet.nco_split_corrected import three_way
    else:
        from rwet.nco_split import three_way

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = femr.models.tokenizer.FEMRTokenizer.from_pretrained(os.path.join(WORK, "tokenizer"))
    task = femr.models.tasks.CLMBRTask(clmbr_vocab_size=VOCAB)
    proc = femr.models.processor.FEMRBatchProcessor(tok, task)
    batches = datasets.load_from_disk(os.path.join(WORK, "batches"))
    batches.set_format("pt")

    with open(os.path.join(WORK, "nco_map.pkl"), "rb") as f:
        ncomap = pickle.load(f)
    with open(os.path.join(WORK, "treatment_tokens.pkl"), "rb") as f:
        treat_tokens = pickle.load(f)

    obj_idx, val_idx, _test_idx, clusters, _W = three_way()
    # nco_map clusters are the token-level panel; align by name
    tok_clusters = ncomap["clusters"]
    name2tokcol = {c: i for i, c in enumerate(tok_clusters)}
    obj_cols = [name2tokcol[clusters[j]] for j in obj_idx if clusters[j] in name2tokcol]
    val_cols = [name2tokcol[clusters[j]] for j in val_idx if clusters[j] in name2tokcol]

    nco_set = set(ncomap["tokens"])
    token2cluster = ncomap["token2cluster"]
    cidx = {c: i for i, c in enumerate(tok_clusters)}
    treat_set = set(treat_tokens.values())
    n_all = len(tok_clusters)

    print(f"[{a.tag}] dev={dev} lam={a.lam} hidden={a.hidden} layers={a.layers} "
          f"lr={a.lr} epochs={a.epochs}", flush=True)
    print(f"[{a.tag}] objective NCOs {len(obj_cols)} | validation NCOs {len(val_cols)}",
          flush=True)

    cfg_t = femr.models.config.FEMRTransformerConfig(
        vocab_size=tok.vocab_size, is_hierarchical=tok.is_hierarchical,
        n_layers=a.layers, hidden_size=a.hidden,
        intermediate_size=a.hidden * 2, n_heads=a.heads)
    cfg = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        cfg_t, task.get_task_config())
    model = femr.models.transformer.FEMRModel(cfg).to(dev)
    model.ln1 = torch.nn.Linear(a.hidden, n_all).to(dev)
    model.ln2 = torch.nn.Linear(a.hidden, a.hidden).to(dev)
    model.train()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.tag}] parameters {n_par:,} ({n_par/1e6:.2f} M)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = torch.nn.BCELoss()

    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev, non_blocking=True)
        if isinstance(x, collections.abc.Mapping):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(move(v) for v in x)
        return x

    n_use = len(batches) if a.max_batches == 0 else min(a.max_batches, len(batches))
    hist = collections.defaultdict(list)
    buf_z, buf_a, buf_w = [], [], []
    t0, step, upd = time.time(), 0, 0

    for ep in range(a.epochs):
        for bi in tqdm(range(n_use), desc=f"[{a.tag}] ep{ep+1}/{a.epochs}", mininterval=30):
            b = move(femr.models.transformer.remove_first_dimension(
                proc.collate([batches[bi]])["batch"]))
            feats = model.transformer(b["transformer"])
            feats = feats.reshape(-1, feats.shape[-1])
            feats = feats[b["transformer"]["label_indices"], :]
            ar_loss, _ = model.task_model(feats, b["task"])
            ar_loss.backward(retain_graph=(a.lam > 0))

            if a.lam > 0:
                lab = b["transformer"]["label_indices"].detach().cpu().numpy()
                tl = b["transformer"]["tokens"].detach().cpu().numpy()[lab] if len(lab) else []
                tpos = [i for i, t in enumerate(tl) if int(t) in treat_set]
                if tpos:
                    j = tpos[0]
                    vec = np.zeros(n_all, dtype=np.float32)
                    for t in set(int(x) for x in tl[j:]) & nco_set:
                        vec[cidx[token2cluster[t]]] = 1.0
                    buf_z.append(feats[j, :])
                    buf_a.append(model.transformer.embed(
                        b["transformer"]["tokens"][lab][j]))
                    buf_w.append(torch.tensor(vec, device=dev))

            step += 1
            if step % a.accum == 0:
                upd += 1
                # Alternating schedule: --penalty-every K applies the negative-
                # control penalty on one update in every K, the other K-1 being
                # plain descent on the next-code loss. The penalty is estimated
                # from an accumulated buffer of rare control events and is the
                # noisiest term in the objective, so imposing it on every update
                # injects that noise into every parameter step. K=1 reproduces
                # the original schedule.
                pen_now = (a.lam > 0) and (upd % a.penalty_every == 0)
                if pen_now and len(buf_z) >= 8:
                    Z, A = torch.stack(buf_z), torch.stack(buf_a)
                    W = torch.stack(buf_w)
                    w_pred, t_pred = model.pred_w(Z), model.pred_t(Z)
                    resid_W, resid_T = W - w_pred, A - t_pred
                    pc_obj = _pc(resid_W[:, obj_cols], resid_T)
                    with torch.no_grad():
                        pc_val = _pc(resid_W[:, val_cols], resid_T)
                    (bce(w_pred, W) + a.lam * pc_obj).backward()
                    hist["pc_obj"].append(float(pc_obj))
                    hist["pc_val"].append(float(pc_val))
                    hist["ar"].append(float(ar_loss))
                    hist["step"].append(step)
                else:
                    # plain descent update: no penalty this step
                    hist["ar"].append(float(ar_loss))
                    hist["step"].append(step)
                opt.step(); opt.zero_grad(set_to_none=True)
                buf_z, buf_a, buf_w = [], [], []

    opt.step(); opt.zero_grad(set_to_none=True)

    out = os.path.join(WORK, f"model_{a.tag}")
    model.save_pretrained(out)

    def tail(key, frac=0.15):
        v = hist.get(key, [])
        if not v:
            return float("nan")
        k = max(1, int(len(v) * frac))
        return float(np.mean(v[-k:]))

    summary = dict(tag=a.tag, penalty=a.penalty, lam=a.lam, hidden=a.hidden, layers=a.layers,
                   lr=a.lr, epochs=a.epochs, params=n_par,
                   pc_obj_final=tail("pc_obj"), pc_val_final=tail("pc_val"),
                   ar_final=tail("ar"), minutes=(time.time() - t0) / 60,
                   penalty_steps=len(hist.get("pc_obj", [])))
    with open(os.path.join(out, "sweep_summary.json"), "w") as f:
        json.dump(dict(summary=summary, hist=dict(hist)), f)

    print(f"\n[{a.tag}] DONE  pc_val={summary['pc_val_final']:.6f}  "
          f"ar={summary['ar_final']:.4f}  {summary['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
