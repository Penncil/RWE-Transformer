"""One plotting style for every cohort, calibrated to the printed page.

The submitted figures came from two toolchains -- the applied analyses were drawn
in R with the OHDSI empirical-calibration routines, the semi-synthetic work in
matplotlib -- and differ in typeface, point size, axis treatment and colour. This
module gives a single set of definitions and a single funnel-plot routine so that
MIMIC-IV and Penn Medicine figures are directly comparable, which they were not
before.

WHY THE TYPE WAS TOO SMALL (Reviewer 1, minor #1). Font sizes here are stated as
the size they should be *on paper*, and are inflated by the factor LaTeX will
later shrink the figure by. The manuscript's \\textwidth is 443.6pt = 6.14in, so a
figure generated 15in wide and included at width=\\textwidth is scaled by 0.41 and
a nominal 9pt label prints at 3.7pt. Call use() with the generated figure width
and every size below lands at its stated value on the page.

    ps.use(11.0)        # figure will be created 11in wide
    fig, ax = plt.subplots(figsize=(11.0, 6.0))

R is not available in this environment, so everything is drawn in matplotlib.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- palette ---------------------------------------------------------------
POINT = "#3B4CC0"        # negative-control estimates
POINT_EDGE = "#1A237E"
NULL_BAND = "#E8A33D"    # acceptance region of the fitted empirical null
NULL_EDGE = "#C8801A"
REFERENCE = "#444444"    # conventional significance boundary
PANEL_BG = "#F5F5F5"
ACCENT = "#c44e52"
NEUTRAL = "#9aa0a6"

# --- sizes, in points AS PRINTED -------------------------------------------
PRINT_WIDTH_IN = 6.14    # \textwidth of RWE-GPTV5.tex (443.58pt / 72.27)
BODY = 8.0               # Reviewer 1 asked for >= 8pt at final print size
TICK = 8.0
TITLE = 9.0
ANNOT = 7.5              # a label, not content; 8pt applies to axis text
ROWLAB = 8.5

_K = 1.0                 # inflation factor, set by use()


def scale() -> float:
    """Current inflation factor, so callers can size their own text."""
    return _K


def pt(size_on_paper: float) -> float:
    """Convert a desired printed point size to a matplotlib size."""
    return size_on_paper * _K


def use(fig_width_in: float = PRINT_WIDTH_IN, print_width_in: float = PRINT_WIDTH_IN):
    """Apply the shared style, inflated so type prints at the sizes above.

    fig_width_in    width the figure will be created at
    print_width_in  width it will occupy on the page (default \\textwidth)
    """
    global _K
    _K = float(fig_width_in) / float(print_width_in)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": pt(BODY),
        "axes.labelsize": pt(BODY),
        "axes.titlesize": pt(TITLE),
        "axes.titleweight": "bold",
        "xtick.labelsize": pt(TICK),
        "ytick.labelsize": pt(TICK),
        "legend.fontsize": pt(TICK),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8 * _K,
        "xtick.major.width": 0.8 * _K,
        "ytick.major.width": 0.8 * _K,
        "xtick.major.size": 3.0 * _K,
        "ytick.major.size": 3.0 * _K,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
    return _K


# Sparse ticks: the submitted panels carried eight labelled ticks across a
# ~1.4in panel, which is most of why they read as dense.
RR_TICKS = np.array([0.25, 1, 4, 10])


def funnel(ax, est, se, mu=None, sigma=None, ease=None, title=None,
           ylab=True, xlim=(0.2, 12), ymax=1.5, annotate=True):
    """Negative-control funnel plot, on the log risk-ratio scale.

    est, se   per-control log risk ratio and its standard error
    mu, sigma parameters of the fitted empirical null; the shaded region is
              |logRR - mu| <= 1.96 sqrt(sigma^2 + se^2), the values a control
              could take without contradicting that null
    The dashed lines are the conventional boundary |logRR| = 1.96 se.
    """
    lo, hi = np.log(xlim[0]), np.log(xlim[1])
    yy = np.linspace(1e-4, ymax, 300)
    ax.set_facecolor(PANEL_BG)

    if mu is not None and sigma is not None and np.isfinite(mu) and np.isfinite(sigma):
        half = 1.96 * np.sqrt(sigma ** 2 + yy ** 2)
        ax.fill_betweenx(yy, np.clip(mu - half, lo, hi), np.clip(mu + half, lo, hi),
                         color=NULL_BAND, alpha=0.85, lw=0, zorder=1)
        ax.plot(np.clip(mu - half, lo, hi), yy, color=NULL_EDGE, lw=1.0 * _K, zorder=3)
        ax.plot(np.clip(mu + half, lo, hi), yy, color=NULL_EDGE, lw=1.0 * _K, zorder=3)

    ax.plot(np.clip(-1.96 * yy, lo, hi), yy, "--", color=REFERENCE,
            lw=1.0 * _K, zorder=4)
    ax.plot(np.clip(1.96 * yy, lo, hi), yy, "--", color=REFERENCE,
            lw=1.0 * _K, zorder=4)
    ax.axvline(0.0, color="black", lw=1.5 * _K, zorder=5)

    est = np.asarray(est, float)
    se = np.asarray(se, float)
    ax.scatter(np.clip(est, lo, hi), np.clip(se, 0, ymax), s=28 * _K ** 2,
               color=POINT, alpha=0.8, edgecolor="white", linewidth=0.4 * _K,
               zorder=6)

    ax.set_xlim(lo, hi)
    # headroom for the EASE label, which sits inside the panel so that it cannot
    # collide with the column title or spill into the neighbouring panel
    ax.set_ylim(0, ymax * 1.34)
    ax.set_xticks(np.log(RR_TICKS))
    ax.set_xticklabels([f"{v:g}" for v in RR_TICKS])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Relative risk", labelpad=2 * _K)
    if ylab:
        ax.set_ylabel("Standard error", labelpad=2 * _K)
    if title:
        ax.set_title(title, pad=6 * _K)
    if annotate and ease is not None and np.isfinite(ease):
        ax.text(0.035, 0.975, f"EASE {ease:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=pt(ANNOT), zorder=20,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#888888",
                          lw=0.6 * _K))
    return ax


def row_label(ax, text, x=-0.40):
    ax.text(x, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=pt(ROWLAB), fontweight="bold", linespacing=1.3)


def finish(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
