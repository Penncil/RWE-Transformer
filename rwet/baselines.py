"""
PyTorch reimplementations of the two neural baselines, following the published
source.

Neither package installs here: both require TensorFlow, which has no
distribution for this Python version. They are therefore reimplemented from
anthem-ai/bcauss (models.py) and SUwonglab/CausalEGM (causalEGM.py), and
validated against the semi-synthetic benchmark before use, so the comparison
rests on demonstrated behaviour rather than on the name attached to the code.

BCAUSS -- Tesei et al., J Biomed Inform 2023;144:104339
    Representation: three dense layers of 200 units.
    Propensity: a single sigmoid unit on the representation.
    Outcome: two heads, each 100 - 100 - 1 with a LINEAR final layer and L2
             regularisation, fitted by squared error; the original treats binary
             outcomes this way, so predictions are clipped to the unit interval
             only when forming risks.
    Loss: factual squared error, plus binary cross-entropy on the propensity,
          plus the auto-balancing term -- covariate means weighted by the
          network's own inverse propensity are required to agree between arms.

CausalEGM -- Liu, Chen and Wong, PNAS 2024;121:e2322376121
    Latent blocks, following causalEGM.py:
        z0  confounder, acts on BOTH treatment and outcome
        z1  acts on the outcome only
        z2  acts on the treatment only
        z3  independent noise
    Outcome network takes (z0, z1, t); treatment network takes (z0, z2). An
    encoder and generator are trained with reconstruction and adversarial terms,
        g_e_loss = g_loss_adv + e_loss_adv
                   + alpha*(l2_loss_v + l2_loss_z) + beta*(l2_loss_x + l2_loss_y)
    and the individual effect is f(z0, z1, 1) - f(z0, z1, 0).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _std(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / (sd + 1e-8)


def _std_like(X, X_eval):
    """Standardise X_eval with the moments of X.

    Used when a fitted network is applied to rows it was not trained on: the
    evaluation rows must be centred and scaled by the training moments, or the
    network sees a different input distribution from the one it was fitted to.
    """
    X = np.asarray(X, float)
    X_eval = np.asarray(X_eval, float)
    if X.ndim == 1:
        X, X_eval = X[:, None], X_eval[:, None]
    mu, sd = X.mean(0), X.std(0)
    return (X_eval - mu) / (sd + 1e-8)


def _stack(d_in, units, d_out, act, final_act=None):
    layers, d = [], d_in
    for u in units:
        layers += [nn.Linear(d, u), act()]
        d = u
    layers += [nn.Linear(d, d_out)]
    if final_act is not None:
        layers += [final_act()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# BCAUSS
# ---------------------------------------------------------------------------
class _BCAUSS(nn.Module):
    def __init__(self, d_in, rep_units=200, head_units=100):
        super().__init__()
        self.rep = _stack(d_in, [rep_units, rep_units], rep_units, nn.ELU,
                          final_act=nn.ELU)
        self.prop = nn.Linear(rep_units, 1)
        self.y0 = _stack(rep_units, [head_units, head_units], 1, nn.ELU)
        self.y1 = _stack(rep_units, [head_units, head_units], 1, nn.ELU)

    def forward(self, x):
        h = self.rep(x)
        return (self.y0(h).squeeze(-1), self.y1(h).squeeze(-1),
                torch.sigmoid(self.prop(h)).squeeze(-1))


def fit_bcauss(X, A, Y, epochs=300, lr=1e-3, b_ratio=1.0, ratio=1.0,
               reg_l2=1e-2, seed=0, clip=0.05, X_eval=None):
    """Left at the untuned default, which is what the reported results use.

    Pass X_eval to obtain counterfactual risks for rows the network was NOT
    fitted on; without it the returned risks are for the fitting rows, as
    before. This is what makes an honest hold-out possible: the outcome heads
    never see the evaluation rows' outcomes, or their covariates.

    A matched eight-point grid was run for both baselines on the simulated
    benchmark (tune_bcauss.py, tune_causalegm.py) to check that CausalEGM's
    advantage was not an artefact of it having been tuned and BCAUSS not. It is
    not: under equal search CausalEGM reaches RMSE 0.068 and BCAUSS 0.143. The
    tuned BCAUSS setting is epochs=2000, b_ratio=1.0, lr=1e-3, which improves it
    from 0.163 to 0.143 -- a real but modest gain that does not change any
    ordering, so the applied results were not re-run for it."""
    torch.manual_seed(seed)
    Xs = torch.tensor(_std(X), dtype=torch.float32, device=DEV)
    t = torch.tensor(np.asarray(A, float), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.asarray(Y, float), dtype=torch.float32, device=DEV)
    m = _BCAUSS(Xs.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=reg_l2 * 1e-2)
    bce = nn.BCELoss()
    for _ in range(epochs):                      # full-batch, as in the original
        opt.zero_grad()
        y0, y1, e = m(Xs)
        e = e.clamp(clip, 1 - clip)
        # factual squared error, each patient through the head it belongs to
        loss = (((1 - t) * (y - y0) ** 2 + t * (y - y1) ** 2).mean()
                + ratio * bce(e, t))
        # auto-balancing: inverse-propensity-weighted covariate means agree
        wt, wc = t / e, (1 - t) / (1 - e)
        mt = (wt[:, None] * Xs).sum(0) / wt.sum().clamp_min(1e-6)
        mc = (wc[:, None] * Xs).sum(0) / wc.sum().clamp_min(1e-6)
        loss = loss + b_ratio * ((mt - mc) ** 2).mean()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        Xo = Xs if X_eval is None else torch.tensor(
            _std_like(X, X_eval), dtype=torch.float32, device=DEV)
        y0, y1, _ = m(Xo)
    return (y0.clamp(0, 1).cpu().numpy(), y1.clamp(0, 1).cpu().numpy())


# ---------------------------------------------------------------------------
# CausalEGM
# ---------------------------------------------------------------------------
class _EGM(nn.Module):
    """z0 confounder (both), z1 outcome only, z2 treatment only, z3 noise."""

    def __init__(self, d_in, z_dims=(3, 3, 6, 6), units=(256, 256, 256)):
        super().__init__()
        self.z_dims = z_dims
        d_z = sum(z_dims)
        lrelu = lambda: nn.LeakyReLU(0.2)                       # noqa: E731
        self.enc = _stack(d_in, list(units), d_z, lrelu)        # e_net
        self.gen = _stack(d_z, list(units), d_in, lrelu)        # g_net
        z0, z1, z2, z3 = z_dims
        self.f_net = _stack(z0 + z1 + 1, [128, 128], 1, lrelu)  # outcome
        self.h_net = _stack(z0 + z2, [128, 128], 1, lrelu)      # treatment

    def split(self, z):
        z0, z1, z2, z3 = self.z_dims
        a = z[:, :z0]
        b = z[:, z0:z0 + z1]
        c = z[:, z0 + z1:z0 + z1 + z2]
        d = z[:, z0 + z1 + z2:]
        return a, b, c, d

    def outcome(self, z0, z1, t):
        return self.f_net(torch.cat([z0, z1, t], 1)).squeeze(-1)

    def forward(self, x, t):
        z = self.enc(x)
        z0, z1, z2, z3 = self.split(z)
        xhat = self.gen(z)
        that = torch.sigmoid(self.h_net(torch.cat([z0, z2], 1))).squeeze(-1)
        yhat = self.outcome(z0, z1, t[:, None])
        return z, xhat, that, yhat


def fit_causalegm(X, A, Y, epochs=2000, lr=1e-3, alpha=1.0, beta=1.0,
                  seed=0, z_dims=(3, 3, 6, 6), X_eval=None):
    """Defaults chosen on the simulated benchmark with known truth
    (tune_causalegm.py): at the shorter schedule the adversarial training had
    not converged and the estimator was worse than no adjustment at all.

    X_eval behaves as in fit_bcauss: the trained encoder and outcome network are
    applied to rows held out of fitting, so a selection rule built on them is
    genuinely out of sample."""
    torch.manual_seed(seed)
    Xs = torch.tensor(_std(X), dtype=torch.float32, device=DEV)
    t = torch.tensor(np.asarray(A, float), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.asarray(Y, float), dtype=torch.float32, device=DEV)
    m = _EGM(Xs.shape[1], z_dims=z_dims).to(DEV)
    d_z = sum(z_dims)
    tanh = lambda: nn.Tanh()                                    # noqa: E731
    dz = _stack(d_z, [256, 256], 1, tanh).to(DEV)               # dz_net
    dv = _stack(Xs.shape[1], [256, 256], 1, tanh).to(DEV)       # dv_net
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(list(dz.parameters()) + list(dv.parameters()),
                             lr=lr, betas=(0.5, 0.9))
    bce_l = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ones = torch.ones(len(Xs), 1, device=DEV)
    zeros = torch.zeros(len(Xs), 1, device=DEV)

    for _ in range(epochs):
        # discriminators separate encoded latent from noise, and generated
        # covariates from observed ones
        with torch.no_grad():
            z = m.enc(Xs)
            xg = m.gen(torch.randn(len(Xs), d_z, device=DEV))
        opt_d.zero_grad()
        ld = (bce_l(dz(torch.randn_like(z)), ones) + bce_l(dz(z), zeros)
              + bce_l(dv(Xs), ones) + bce_l(dv(xg), zeros))
        ld.backward()
        opt_d.step()

        opt.zero_grad()
        z, xhat, that, yhat = m(Xs, t)
        zr = torch.randn(len(Xs), d_z, device=DEV)
        xr = m.gen(zr)
        zrec = m.enc(xr)
        loss = (bce_l(dz(z), ones) + bce_l(dv(xr), ones)          # adversarial
                + alpha * (mse(xhat, Xs) + mse(zrec, zr))         # roundtrip
                + beta * (nn.functional.binary_cross_entropy(
                    that.clamp(1e-6, 1 - 1e-6), t) + mse(yhat, y)))
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        Xo = Xs if X_eval is None else torch.tensor(
            _std_like(X, X_eval), dtype=torch.float32, device=DEV)
        z = m.enc(Xo)
        z0, z1, _z2, _z3 = m.split(z)
        n = len(Xo)
        y1 = m.outcome(z0, z1, torch.ones(n, 1, device=DEV)).clamp(0, 1)
        y0 = m.outcome(z0, z1, torch.zeros(n, 1, device=DEV)).clamp(0, 1)
    return y0.cpu().numpy(), y1.cpu().numpy()


# ---------------------------------------------------------------------------
def point_estimate(y0, y1):
    """Log risk ratio from predicted counterfactual risks."""
    m1, m0 = max(float(np.mean(y1)), 1e-10), max(float(np.mean(y0)), 1e-10)
    return float(np.log(m1 / m0))


def point_estimates_by_population(y0, y1, A):
    """Log risk ratio for the ATE, ATT and ATC populations.

    Both counterfactual risks are predicted for every patient, so these three
    estimands differ only in the population the contrast is averaged over --
    there is nothing further to fit. This is how the submitted analysis obtained
    ATT and ATC for these baselines, and it is why all three are available here.
    """
    A = np.asarray(A, float)
    y0, y1 = np.asarray(y0, float).ravel(), np.asarray(y1, float).ravel()
    out = {}
    for name, m in (("ATE", np.ones(len(A), dtype=bool)),
                    ("ATT", A == 1), ("ATC", A == 0)):
        if m.sum() < 1:
            out[name] = np.nan
            continue
        m1 = max(float(np.mean(y1[m])), 1e-10)
        m0 = max(float(np.mean(y0[m])), 1e-10)
        out[name] = float(np.log(m1 / m0))
    return out


def bootstrap_refit_all(kind, X, A, Y, n_boot=10, seed=0, **kw):
    """ATE/ATT/ATC point estimates and refit standard errors in one pass.

    Each bootstrap replicate refits the network once and yields all three
    estimands, so the three cost no more than one.
    """
    fit = fit_bcauss if kind == "bcauss" else fit_causalegm
    X = np.asarray(X, float)
    A = np.asarray(A, float)
    Y = np.asarray(Y, float)
    y0, y1 = fit(X, A, Y, seed=seed, **kw)
    est = point_estimates_by_population(y0, y1, A)
    tau = np.asarray(y1, float).ravel() - np.asarray(y0, float).ravel()

    rng = np.random.default_rng(seed)
    n, reps = len(A), {"ATE": [], "ATT": [], "ATC": []}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if A[i].sum() < 20 or Y[i].sum() < 10:
            continue
        try:
            b0, b1 = fit(X[i], A[i], Y[i], seed=seed + 1000 + b, **kw)
            v = point_estimates_by_population(b0, b1, A[i])
        except Exception:
            continue
        for k, val in v.items():
            if np.isfinite(val):
                reps[k].append(val)
    # 3 surviving replicates is the minimum for a usable sd; with n_boot as low
    # as 4 a single skipped resample would otherwise discard the control
    se = {k: (float(np.std(v, ddof=1)) if len(v) >= 3 else np.nan)
          for k, v in reps.items()}
    return est, se, tau, {k: len(v) for k, v in reps.items()}


def effect_from_counterfactuals(y0, y1, n_boot=200, seed=0):
    """Resample patients holding the fitted predictions fixed.

    This is NOT a valid standard error and is retained only to demonstrate the
    failure mode. Holding the predictions fixed treats a fitted model as if it
    were known, so the interval reflects the variability of averaging and not
    the uncertainty in the quantity being averaged. On the applied cohort it
    returns values near 0.003-0.01 against a crude two-arm floor of 0.07, which
    is the same order-of-magnitude understatement this revision documents in the
    submitted pipeline. Use bootstrap_refit for inference.
    """
    est = point_estimate(y0, y1)
    rng = np.random.default_rng(seed)
    n, b = len(y0), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a1, a0 = float(np.mean(y1[i])), float(np.mean(y0[i]))
        if a1 > 0 and a0 > 0:
            b.append(np.log(a1 / a0))
    se = float(np.std(b, ddof=1)) if len(b) > 5 else np.nan
    return est, se


def bootstrap_refit(kind, X, A, Y, n_boot=20, seed=0, **kw):
    """Standard error from resampling patients and REFITTING the model.

    Each replicate draws a bootstrap sample and re-estimates the network on it,
    so the spread of the resulting estimates carries the uncertainty in the
    fitted model as well as in the sample. This is the expensive but honest
    version.
    """
    fit = fit_bcauss if kind == "bcauss" else fit_causalegm
    y0, y1 = fit(X, A, Y, seed=seed, **kw)
    est = point_estimate(y0, y1)
    X = np.asarray(X, float)
    A = np.asarray(A, float)
    Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    n, reps = len(A), []
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if A[i].sum() < 20 or Y[i].sum() < 10:
            continue
        try:
            b0, b1 = fit(X[i], A[i], Y[i], seed=seed + 1000 + b, **kw)
            v = point_estimate(b0, b1)
            if np.isfinite(v):
                reps.append(v)
        except Exception:
            continue
    se = float(np.std(reps, ddof=1)) if len(reps) > 5 else np.nan
    return est, se, len(reps)


def run_baseline(kind, X, A, Y, seed=0, n_boot=20, refit=True, **kw):
    if refit:
        est, se, nb = bootstrap_refit(kind, X, A, Y, n_boot=n_boot, seed=seed, **kw)
        y0, y1 = (fit_bcauss if kind == "bcauss" else fit_causalegm)(
            X, A, Y, seed=seed, **kw)
    else:
        y0, y1 = (fit_bcauss if kind == "bcauss" else fit_causalegm)(
            X, A, Y, seed=seed, **kw)
        est, se = effect_from_counterfactuals(y0, y1, seed=seed)
        nb = 0
    return dict(logRR=est, se=se, n_boot=nb, mu1=float(np.mean(y1)),
                mu0=float(np.mean(y0)), ate=float(np.mean(y1 - y0)),
                y0=y0, y1=y1)
