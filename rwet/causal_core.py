"""
Cross-fitted causal estimation core for RWE-Transformer downstream analysis.

What is different from the current pipeline
-------------------------------------------
1. CROSS-FITTING (the main correctness fix). Nuisance functions e(Z) and mu_a(Z)
   are fitted on K-1 folds and evaluated on the held-out fold. Fitting a flexible
   model on the same 6,607 patients you then evaluate it on drives the residuals
   (Y - mu_hat) toward zero, which biases the point estimate and shrinks the
   variance. With a 64-dimensional representation this is not a small effect.
   Cross-fitting is the standard remedy (Chernozhukov et al., double machine
   learning) and costs nothing but a K-fold loop.

2. OVERLAP DIAGNOSTICS. Propensity truncation is reported, not silent, and the
   effective sample size after weighting is returned so that a near-violation of
   positivity is visible rather than buried.

3. EMPIRICAL CALIBRATION with the NCO empirical null fitted as
   psi_k ~ N(mu, sigma^2 + s_k^2), so NCO sampling error is not absorbed into
   sigma^2, and computed on a HELD-OUT NCO panel so the diagnostic is not
   evaluated on the quantity that was optimised.

Standard errors come from the empirical influence function. Under cross-fitting
these are asymptotically valid without a bootstrap, which also makes the whole
pipeline an order of magnitude cheaper to run.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PS_CLIP = 0.02
C_REG = 0.5


# ---------------------------------------------------------------------------
# nuisance estimation
# ---------------------------------------------------------------------------
def _standardise(F):
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    mu, sd = F.mean(0), F.std(0)
    return (F - mu) / (sd + 1e-8)


def crossfit_nuisance(F, A, Y, n_splits=5, seed=0, C=C_REG):
    """Out-of-fold e(Z), mu1(Z), mu0(Z).

    Every prediction for patient i comes from a model that never saw i.
    """
    F = _standardise(F)
    n = len(A)
    e = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)

    # stratify on the (A, Y) cell so rare cells are spread across folds
    strat = (A.astype(int) * 2 + (Y > 0).astype(int))
    counts = np.bincount(strat, minlength=4)
    k = int(min(n_splits, max(2, counts[counts > 0].min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for tr, te in skf.split(F, strat):
        ps = LogisticRegression(max_iter=5000, C=C, random_state=seed)
        ps.fit(F[tr], A[tr])
        e[te] = ps.predict_proba(F[te])[:, 1]

        for a, dest in ((1, mu1), (0, mu0)):
            m = tr[A[tr] == a]
            ya = Y[m]
            if len(ya) == 0 or ya.sum() == 0 or ya.sum() == len(ya):
                dest[te] = ya.mean() if len(ya) else 0.0
                continue
            om = LogisticRegression(max_iter=5000, C=C, random_state=seed)
            om.fit(F[m], ya)
            dest[te] = om.predict_proba(F[te])[:, 1]

    e = np.clip(e, PS_CLIP, 1 - PS_CLIP)
    return e, mu1, mu0


# ---------------------------------------------------------------------------
# AIPW with influence-function inference
# ---------------------------------------------------------------------------
def aipw_crossfit(F, A, Y, n_splits=5, seed=0, C=C_REG, return_diag=False):
    n = len(A)
    e, mu1, mu0 = crossfit_nuisance(F, A, Y, n_splits, seed, C)

    psi1 = mu1 + A * (Y - mu1) / e
    psi0 = mu0 + (1 - A) * (Y - mu0) / (1 - e)

    out = {}

    d = psi1 - psi0
    out["ATE"] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)))

    m1, m0 = max(psi1.mean(), 1e-8), max(psi0.mean(), 1e-8)
    ic = (psi1 - psi1.mean()) / m1 - (psi0 - psi0.mean()) / m0
    out["logRR"] = (float(np.log(m1 / m0)), float(ic.std(ddof=1) / np.sqrt(n)))

    pA = max(A.mean(), 1e-8)
    att = (A * (Y - mu0) - (1 - A) * e / (1 - e) * (Y - mu0)) / pA
    out["ATT"] = (float(att.mean()), float(att.std(ddof=1) / np.sqrt(n)))

    pC = max(1 - A.mean(), 1e-8)
    atc = (A * (1 - e) / e * (Y - mu1) - (1 - A) * (Y - mu1)) / pC
    out["ATC"] = (float(atc.mean()), float(atc.std(ddof=1) / np.sqrt(n)))

    if return_diag:
        w = np.where(A == 1, 1 / e, 1 / (1 - e))
        out["_diag"] = dict(
            ps_min=float(e.min()), ps_max=float(e.max()),
            frac_clipped=float(np.mean((e <= PS_CLIP + 1e-12) |
                                       (e >= 1 - PS_CLIP - 1e-12))),
            ess=float(w.sum() ** 2 / (w ** 2).sum()),   # Kish effective sample size
            n=n,
        )
    return out


def aipw_insample(F, A, Y, seed=0, C=C_REG):
    """The current (non-cross-fitted) estimator, kept for comparison."""
    F = _standardise(F)
    n = len(A)
    ps = LogisticRegression(max_iter=5000, C=C, random_state=seed).fit(F, A)
    e = np.clip(ps.predict_proba(F)[:, 1], PS_CLIP, 1 - PS_CLIP)

    def arm(a):
        m = A == a
        ya = Y[m]
        if ya.sum() == 0 or ya.sum() == m.sum():
            return np.full(n, ya.mean() if len(ya) else 0.0)
        return LogisticRegression(max_iter=5000, C=C, random_state=seed) \
            .fit(F[m], ya).predict_proba(F)[:, 1]

    mu1, mu0 = arm(1), arm(0)
    psi1 = mu1 + A * (Y - mu1) / e
    psi0 = mu0 + (1 - A) * (Y - mu0) / (1 - e)
    m1, m0 = max(psi1.mean(), 1e-8), max(psi0.mean(), 1e-8)
    ic = (psi1 - psi1.mean()) / m1 - (psi0 - psi0.mean()) / m0
    d = psi1 - psi0
    return {"ATE": (float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))),
            "logRR": (float(np.log(m1 / m0)),
                      float(ic.std(ddof=1) / np.sqrt(n)))}


# ---------------------------------------------------------------------------
# empirical null / EASE
# ---------------------------------------------------------------------------
def fit_empirical_null(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0) & (np.abs(est) < 10)
    est, se = est[ok], se[ok]
    if len(est) < 3:
        return np.nan, np.nan, len(est)

    def nll(t):
        v = np.exp(2 * t[1]) + se ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * v) + (est - t[0]) ** 2 / v)

    r = optimize.minimize(nll, [est.mean(), np.log(est.std() + 1e-2)],
                          method="Nelder-Mead",
                          options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
    return float(r.x[0]), float(np.exp(r.x[1])), len(est)


def ease_from_null(mu, sigma):
    """E|psi| for psi ~ N(mu, sigma^2), the folded-normal mean.

    The sigma -> 0 limit is |mu|: the null collapses to a point mass at mu. That
    limit has to be taken explicitly, because sigma can be small enough that
    sigma**2 underflows to exactly zero and the closed form divides by it. This
    happens on bootstrap resamples where the control estimates are fully
    explained by their own sampling error.
    """
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    if sigma < 1e-8:                      # below this, sigma**2 is unreliable
        return float(abs(mu))
    return float(sigma * np.sqrt(2 / np.pi) * np.exp(-mu ** 2 / (2 * sigma ** 2))
                 + mu * (1 - 2 * stats.norm.cdf(-mu / sigma)))


def nco_null(F, A, W, min_events=15, n_splits=5, seed=0, crossfit=True):
    """Run the pipeline on each NCO, then fit the empirical null."""
    est, se = [], []
    for j in range(W.shape[1]):
        w = np.asarray(W[:, j], float)
        if w.sum() < min_events or w.sum() > len(w) - min_events:
            continue
        try:
            r = (aipw_crossfit(F, A, w, n_splits, seed) if crossfit
                 else aipw_insample(F, A, w, seed))
            if np.isfinite(r["logRR"][0]) and np.isfinite(r["logRR"][1]):
                est.append(r["logRR"][0])
                se.append(r["logRR"][1])
        except Exception:
            continue
    mu, sigma, k = fit_empirical_null(est, se)
    return dict(mu=mu, sigma=sigma, k=k, ease=ease_from_null(mu, sigma),
                estimates=np.asarray(est), ses=np.asarray(se))


def calibrate(estimate, se, mu, sigma):
    """Empirical calibration: shift by the null mean, widen by the null sd."""
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return np.nan, np.nan
    return estimate - mu, float(np.sqrt(se ** 2 + sigma ** 2))
