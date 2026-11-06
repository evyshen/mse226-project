"""
MS&E 226 Project Part 1 — Step 1: duplicate check and 80/20 holdout split.

Reproduces the exact development/holdout split used in this project.

Input:  data/NH_ProviderInfo_Oct2026.csv
        CMS Provider Data Catalog, "Nursing Homes Including Rehab Services:
        Provider Information", processing date 2026-10-01
        https://data.cms.gov/provider-data/dataset/4pq5-n9py
Output: data/dev.csv        80% development set (all Part 1 exploration and modeling)
        data/holdout.csv    20% holdout (untouched until Part 2)
        data/split_ccns.csv facility ID -> split label

Usage (from the project folder):
    python 01_dedup_and_split.py

Environment used: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1.

How reproducibility is guaranteed:
  * The split is a seeded random shuffle (random_state=226). scikit-learn's
    train_test_split draws from numpy's legacy RandomState, whose output for a
    given seed is fixed across numpy/sklearn versions.
  * A seeded shuffle reproduces only if the input rows are identical and in
    the same order. CMS refreshes this file monthly, so the script checks the
    SHA-256 fingerprint of the raw file and refuses to run on a different one.
  * After splitting, it checks a fingerprint of the holdout's facility IDs, so
    anyone rerunning can confirm they got the identical 2,938 holdout rows.
  * data/split_ccns.csv records the assignment of every facility, so the split
    can also be rebuilt from IDs alone without rerunning the shuffle.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

SEED = 226
HOLDOUT_FRAC = 0.20
ID = "CMS Certification Number (CCN)"

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW = DATA_DIR / "NH_ProviderInfo_Oct2026.csv"

# Fingerprints from the original run (2026-11-04)
EXPECTED_RAW_SHA256 = "ab70309ae6f0ee86b19fe9acb61a6d2953f970d3b09e80bb9b090e4b19ce35a2"
EXPECTED_HOLDOUT_SHA256 = "e4eeef335d68739e57d27c705a0dc9ce4fc2e07f3477efc369b56298ad6b2f02"
EXPECTED_N = {"raw": 14686, "dev": 11748, "holdout": 2938}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids):
    """Fingerprint of a set of facility IDs, independent of their order."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# --- 0. Confirm this is the same raw file -----------------------------------
raw_hash = sha256_file(RAW)
if raw_hash != EXPECTED_RAW_SHA256:
    sys.exit(
        "Raw file does not match the original (CMS updates it monthly).\n"
        f"  expected {EXPECTED_RAW_SHA256}\n  got      {raw_hash}\n"
        "Use the October 2026 file, or rebuild the split from data/split_ccns.csv."
    )

# Read the facility ID as text: some CCNs contain letters or leading zeros
# (e.g. '075339', '25A197') that would be mangled if read as numbers.
df = pd.read_csv(RAW, low_memory=False, dtype={ID: str})
assert len(df) == EXPECTED_N["raw"]

# --- 1. Duplicate checks ----------------------------------------------------
exact_dups = df.duplicated().sum()
ccn_dups = df[ID].duplicated().sum()
name_addr_dups = df.duplicated(subset=["Provider Name", "Provider Address", "City/Town", "State"]).sum()

df = df.drop_duplicates()
df = df.drop_duplicates(subset=[ID], keep="first")

# --- 2. Seeded 80/20 split --------------------------------------------------
dev, holdout = train_test_split(df, test_size=HOLDOUT_FRAC, random_state=SEED, shuffle=True)

# --- 3. Verify ----------------------------------------------------------------
assert len(dev) == EXPECTED_N["dev"] and len(holdout) == EXPECTED_N["holdout"]
assert set(dev[ID]).isdisjoint(holdout[ID])
holdout_hash = sha256_ids(holdout[ID])
if EXPECTED_HOLDOUT_SHA256 is not None:
    assert holdout_hash == EXPECTED_HOLDOUT_SHA256, "Holdout differs from the original split."

# --- 4. Save ----------------------------------------------------------------
dev.to_csv(DATA_DIR / "dev.csv", index=False)
holdout.to_csv(DATA_DIR / "holdout.csv", index=False)
pd.concat([
    pd.DataFrame({ID: dev[ID], "split": "dev"}),
    pd.DataFrame({ID: holdout[ID], "split": "holdout"}),
]).sort_values(ID).to_csv(DATA_DIR / "split_ccns.csv", index=False)

print(f"python {sys.version.split()[0]}, pandas {pd.__version__}, "
      f"numpy {np.__version__}, scikit-learn {sklearn.__version__}")
print(f"raw rows: {EXPECTED_N['raw']}  (sha256 {raw_hash[:12]}…, verified)")
print(f"exact duplicate rows: {exact_dups}; duplicate CCNs: {ccn_dups}; "
      f"duplicate name+address: {name_addr_dups}")
print(f"rows after de-duplication: {len(df)}")
print(f"seed: {SEED}; dev: {len(dev)}; holdout: {len(holdout)}")
print(f"holdout ID fingerprint: {holdout_hash}")
