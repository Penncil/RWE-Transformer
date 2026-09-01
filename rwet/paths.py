"""Where the data lives.

No path in this repository is absolute. Each root below is read from an
environment variable, falling back to a directory beside the repository, so the
code runs unchanged on any machine once the data is in place.

    RWET_DATA     parent of the cohort directories        default: ../rwe-data
    RWET_WORK     trained models and extracted features   default: ./work
    RWET_RESULTS  analysis outputs                        default: ./results
    RWET_FIGURES  generated figures                       default: ./figures

None of the data is distributed with this repository; see the README for what
each root must contain and how to obtain it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _root(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, default)).expanduser().resolve()


DATA = _root("RWET_DATA", ROOT.parent / "rwe-data")
WORK = _root("RWET_WORK", ROOT / "work")
RESULTS = _root("RWET_RESULTS", ROOT / "results")
FIGURES = _root("RWET_FIGURES", ROOT / "figures")

MIMIC = DATA / "mimic"
AD_DATA = DATA / "ad-deid"
AD_OUTPUTS = AD_DATA / "ad_all_drugs"
AD_COHORTS = AD_DATA / "data_for_bingyu_300"
VOCABULARY = DATA / "omop-vocabulary"


def require(p: Path, what: str) -> Path:
    """Fail with an instruction rather than a stack trace."""
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found at {p}.\n"
            f"Set the corresponding RWET_* environment variable, or place the "
            f"data there. See README.md, 'Obtaining the data'."
        )
    return p


def ad_cohort(drug: str) -> Path:
    """Per-drug Penn Medicine analysis table, e.g. ad_cohort('6918')."""
    return require(AD_COHORTS / f"{drug}_data.csv",
                   f"Penn Medicine cohort for drug {drug}")


for _d in (WORK, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
