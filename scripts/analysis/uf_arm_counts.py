"""OneFlorida+ treated and comparator counts, derived from what survives.

Reviewer 2 asked for treated and control counts for this site; the submitted
version reported only the 39,510 total. Of the OneFlorida+ extract only
`person.csv` and `drug_exposure.csv` remain, but that is enough to count
exposures, which is what the request needs.

Method. Drug codes in `drug_source_value` are RxNorm identifiers for clinical
drugs, so they are resolved to names through the OMOP vocabulary and classified
by ingredient. The exposure rule follows the Methods: at least two prescriptions
with the last falling more than 30 days after the first.

What this cannot do. There is no condition table for this site, so the MCI
eligibility criteria -- an MCI diagnosis, age over 50 at diagnosis, a year of
baseline, no prior AD -- cannot be applied, and progression to Alzheimer's
disease cannot be counted at all. The counts below are therefore exposure counts
within the 39,510-patient extract, and are an upper bound on the analytic arms.

    python uf_arm_counts.py
"""
from __future__ import annotations

import os

from rwet import paths
import re

import pandas as pd

D = (str(paths.AD_OUTPUTS) +
     r"\drug_6918\ITE")
VOCAB = str(paths.VOCABULARY / "CONCEPT.csv")
OUT = str(paths.RESULTS)
MIN_SPAN_DAYS = 30

CLASSES = {
    "metoprolol": re.compile(r"\bmetoprolol\b", re.I),
    "folic acid": re.compile(r"\bfolic acid\b|\bfolate\b", re.I),
    "vitamin B12": re.compile(r"\bcyanocobalamin\b|\bhydroxocobalamin\b"
                              r"|vitamin B ?12", re.I),
}


def main():
    print("scanning drug_exposure ...", flush=True)
    codes, recs = set(), []
    for ch in pd.read_csv(os.path.join(D, "drug_exposure.csv"), chunksize=500_000,
                          low_memory=False,
                          usecols=["person_id", "drug_source_value",
                                   "drug_exposure_start_date"]):
        ch = ch[ch.person_id != "--"].dropna(subset=["drug_source_value"])
        ch["code"] = (ch.drug_source_value.astype(str)
                      .str.replace(r"\.0$", "", regex=True))
        codes.update(ch.code.unique())
        recs.append(ch[["person_id", "code", "drug_exposure_start_date"]])
    ex = pd.concat(recs, ignore_index=True)
    print(f"  {len(ex):,} exposures, {ex.person_id.nunique():,} persons, "
          f"{len(codes):,} distinct codes")

    print("resolving codes through the OMOP vocabulary ...", flush=True)
    name = {}
    for ch in pd.read_csv(VOCAB, sep="\t", chunksize=1_000_000, low_memory=False,
                          usecols=["concept_name", "vocabulary_id",
                                   "concept_code"], on_bad_lines="skip"):
        ch = ch[(ch.vocabulary_id == "RxNorm")
                & ch.concept_code.astype(str).isin(codes)]
        for c, nm in zip(ch.concept_code.astype(str), ch.concept_name):
            name.setdefault(c, nm)
    print(f"  resolved {len(name):,} of {len(codes):,} codes")

    ex["name"] = ex.code.map(name)
    ex["cls"] = None
    for cls, pat in CLASSES.items():
        hit = ex.name.notna() & ex.name.str.contains(pat, na=False)
        ex.loc[hit & ex.cls.isna(), "cls"] = cls

    ex["date"] = pd.to_datetime(ex.drug_exposure_start_date, errors="coerce")
    sub = ex[ex.cls.notna() & ex.date.notna()]
    print("\nexposures by class (before the two-prescription rule):")
    print(sub.groupby("cls").agg(records=("person_id", "size"),
                                 persons=("person_id", "nunique")).to_string())

    g = sub.groupby(["cls", "person_id"])["date"]
    span = (g.max() - g.min()).dt.days
    cnt = g.size()
    qualifies = (cnt >= 2) & (span > MIN_SPAN_DAYS)
    keep = qualifies[qualifies].reset_index()

    print(f"\napplying the rule: >=2 prescriptions spanning > {MIN_SPAN_DAYS} days")
    res = keep.groupby("cls")["person_id"].nunique()
    print(res.to_string())

    met = set(keep.loc[keep.cls == "metoprolol", "person_id"])
    vit = set(keep.loc[keep.cls.isin(["folic acid", "vitamin B12"]),
                       "person_id"])
    print(f"\nmetoprolol initiators           {len(met):,}")
    print(f"folic acid or B12 comparators   {len(vit):,}")
    print(f"  of which also on metoprolol   {len(met & vit):,}"
          "   (excluded from the comparator arm in the analysis)")
    print(f"comparators excluding overlap   {len(vit - met):,}")

    pd.DataFrame([{"arm": "metoprolol", "n": len(met)},
                  {"arm": "folic acid or B12", "n": len(vit)},
                  {"arm": "both", "n": len(met & vit)},
                  {"arm": "comparator excluding overlap", "n": len(vit - met)}]
                 ).to_csv(os.path.join(OUT, "uf_arm_counts.csv"), index=False)
    print(f"\n-> {os.path.join(OUT, 'uf_arm_counts.csv')}")


if __name__ == "__main__":
    main()
