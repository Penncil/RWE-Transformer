"""
Individualised treatment effect (ITE) estimation and negative-control diagnostics.

Why this module exists
----------------------
The submitted analysis reported an ITE column whose EASE was exactly 0.00 for
three of five methods. An EASE of exactly zero is not attainable from a fitted
empirical null -- it requires mu = sigma = 0 -- and inspection of the original
code path shows why it happened: the "ITE" was `torch.round(w_pred)`, i.e. the
rounded predicted probability of each negative control outcome. For outcomes
with prevalence below 0.5 that rounds to 0 for every patient in both arms, so
the estimated individual effect is identically zero by construction. The
estimator therefore passed the diagnostic by being uninformative, not by being
unbiased.

What replaces it
----------------
tau(x) = E[Y(1) - Y(0) | X = x] on the risk-difference scale, estimated with the
cross-fitted doubly-robust learner (Kennedy, 2023):

  stage 1   cross-fitted nuisances e(z), mu_1(z), mu_0(z)
  stage 2   regress the DR pseudo-outcome

              phi_i = mu_1(z_i) - mu_0(z_i)
                      + a_i (y_i - mu_1(z_i)) / e(z_i)
                      - (1 - a_i)(y_i - mu_0(z_i)) / (1 - e(z_i))

            on z, again cross-fitted, so tau_hat(z_i) never sees patient i.

E[phi | Z = z] = tau(z) whenever either nuisance is correct, so the second-stage
regression targets the right object under the same conditions that make the AIPW
point estimate doubly robust.

The negative-control diagnostic for ITE
---------------------------------------
A negative control outcome is not caused by treatment, so its individual effect
is zero *for every patient*, not merely on average. That is a far stronger null
than the one the ATE diagnostic uses, and it yields two diagnostics that the
ATE-level EASE cannot express:

  EASE-ITE    mean absolute estimated individual effect, pooled over patients
              and controls. The individual-level analogue of EASE; the typical
              spurious individual effect the pipeline reports for an outcome the
              treatment cannot influence. Units: risk difference.

  spurious    within-control standard deviation of tau_hat, averaged over
  heterogeneity
              controls. A pipeline that reports patient-to-patient variation in
              the effect of a treatment on an outcome it cannot cause is
              manufacturing heterogeneity. Zero is the correct answer.

Both are reported with a bootstrap interval over the control panel, because with
a panel this small a point estimate on its own would overstate the precision.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from rwet.causal_core import PS_CLIP, C_REG, _standardise, crossfit_nuisance  # noqa: F401

ALPHAS = np.logspace(-2, 4, 25)


# ---------------------------------------------------------------------------
# DR-learner
# ---------------------------------------------------------------------------
def dr_pseudo_outcome(F, A, Y, n_splits=5, seed=0, C=C_REG):
    """Cross-fitted doubly-robust pseudo-outcome. E[phi | Z] = tau(Z)."""
    e, mu1, mu0 = crossfit_nuisance(F, A, Y, n_splits, seed, C)
    phi = (mu1 - mu0
           + A * (Y - mu1) / e
           - (1 - A) * (Y - mu0) / (1 - e))
    return phi, e, mu1, mu0


def dr_learner(F, A, Y, n_splits=5, seed=0, C=C_REG, return_phi=False):
    """Out-of-fold individual effect estimates tau_hat(z_i).

    Both stages are cross-fitted, so tau_hat(z_i) is produced by models that
    never saw patient i. Without this the second stage interpolates its own
    pseudo-outcome and the apparent heterogeneity is mostly overfitting.
    """
    F = _standardise(F)
    n = len(A)
    phi, e, mu1, mu0 = dr_pseudo_outcome(F, A, Y, n_splits, seed, C)

    tau = np.full(n, np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(F):
        m = RidgeCV(alphas=ALPHAS).fit(F[tr], phi[tr])
        tau[te] = m.predict(F[te])

    if return_phi:
        return tau, phi
    return tau


# ---------------------------------------------------------------------------
# negative-control diagnostics for ITE
# ---------------------------------------------------------------------------
def ite_nco_diagnostics(F, A, W, min_events=25, n_splits=5, seed=0,
                        n_boot=2000, clusters=None):
    """Individual-level systematic error over a panel of negative controls.

    Under a valid negative control the individual effect is zero for every
    patient, so every departure of tau_hat from zero is systematic error.
    """
    per = []
    for j in range(W.shape[1]):
        w = np.asarray(W[:, j], float)
        if w.sum() < min_events or w.sum() > len(w) - min_events:
            continue
        try:
            tau = dr_learner(F, A, w, n_splits=n_splits, seed=seed)
        except Exception:
            continue
        if not np.all(np.isfinite(tau)):
            continue
        per.append(dict(
            cluster=(clusters[j] if clusters is not None else str(j)),
            n_events=float(w.sum()),
            mean_abs=float(np.mean(np.abs(tau))),   # EASE-ITE contribution
            sd=float(np.std(tau, ddof=1)),          # spurious heterogeneity
            mean=float(np.mean(tau)),               # collapses to the ATE
            q10=float(np.quantile(tau, 0.10)),
            q90=float(np.quantile(tau, 0.90)),
        ))

    if not per:
        return dict(k=0, ease_ite=np.nan, het=np.nan, per_nco=[])

    ma = np.array([d["mean_abs"] for d in per])
    sd = np.array([d["sd"] for d in per])

    rng = np.random.default_rng(seed)
    b_ease, b_het = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ma), len(ma))
        b_ease.append(ma[idx].mean())
        b_het.append(sd[idx].mean())
    lo, hi = np.percentile(b_ease, [2.5, 97.5])
    hlo, hhi = np.percentile(b_het, [2.5, 97.5])

    return dict(k=len(per),
                ease_ite=float(ma.mean()), ease_lo=float(lo), ease_hi=float(hi),
                het=float(sd.mean()), het_lo=float(hlo), het_hi=float(hhi),
                per_nco=per)


# ---------------------------------------------------------------------------
# evaluation against known ground truth (simulation only)
# ---------------------------------------------------------------------------
def ite_truth_metrics(tau_hat, tau_true):
    """PEHE and the ranking / calibration quality that matters for targeting."""
    from scipy import stats as _st
    tau_hat = np.asarray(tau_hat, float)
    tau_true = np.asarray(tau_true, float)
    ok = np.isfinite(tau_hat) & np.isfinite(tau_true)
    tau_hat, tau_true = tau_hat[ok], tau_true[ok]

    pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    bias = float(np.mean(tau_hat - tau_true))
    sp = float(_st.spearmanr(tau_hat, tau_true).statistic)
    pr = float(np.corrcoef(tau_hat, tau_true)[0, 1])

    # calibration slope: regress truth on estimate. 1.0 is perfect.
    v = np.var(tau_hat)
    slope = float(np.cov(tau_true, tau_hat)[0, 1] / v) if v > 1e-14 else np.nan

    # decile calibration -- do the patients we rank as most-helped actually
    # benefit most?  This is the quantity a targeting policy depends on.
    q = np.quantile(tau_hat, np.linspace(0, 1, 11))
    q[0] -= 1e-9
    grp = np.clip(np.digitize(tau_hat, q[1:-1]), 0, 9)
    gate_hat = np.array([tau_hat[grp == g].mean() for g in range(10)])
    gate_true = np.array([tau_true[grp == g].mean() for g in range(10)])

    # value of targeting the top 20% by estimated benefit, against the truth
    top = tau_hat <= np.quantile(tau_hat, 0.20)   # most negative = most helped
    top_true = float(tau_true[top].mean())
    best_true = float(np.sort(tau_true)[:max(1, int(0.20 * len(tau_true)))].mean())

    return dict(pehe=pehe, bias=bias, spearman=sp, pearson=pr,
                cal_slope=slope, sd_hat=float(np.std(tau_hat)),
                sd_true=float(np.std(tau_true)),
                gate_hat=gate_hat, gate_true=gate_true,
                top20_true_benefit=top_true, top20_oracle_benefit=best_true)
