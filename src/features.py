"""Feature construction and the two feature sets.

Every transformation here is row-wise (it uses only that facility's own values and
fixed constants), so it can be applied before CV without leaking information across
folds. Anything estimated from data (imputation, scaling, encoding) lives in
pipeline.py and is fit inside each training fold.
"""
import numpy as np
import pandas as pd

from .config import ID, OUTCOME, REFERENCE_DATE

# Columns computed from (or contemporaneous with) the inspection we predict -> never used.
LEAKAGE = [
    c for c in [
        "Rating Cycle 1 Standard Survey Health Date",
        "Rating Cycle 1 Standard Survey Footnote",
        "Rating Cycle 1 Number of Standard Health Deficiencies",
        "Rating Cycle 1 Number of Complaint Health Deficiencies",
        "Rating Cycle 1 Health Deficiency Score",
        "Rating Cycle 1 Number of Health Revisits",
        "Rating Cycle 1 Health Revisit Score",
        "Rating Cycle 1 Total Health Score",
        "Total Weighted Health Survey Score",
        "Overall Rating", "Health Inspection Rating",
        "Chain Average Overall 5-star Rating", "Chain Average Health Inspection Rating",
        "Abuse Icon", "Special Focus Status", "High Performing Icon",
        "Most Recent Health Inspection More Than 2 Years Ago",
        "Number of Fines", "Total Amount of Fines in Dollars",
        "Number of Payment Denials", "Total Number of Penalties",
        "Number of Citations from Infection Control Inspections",
    ]
]

# ---- Feature set 1: facility profile (no inspection history) ----------------
FS1_NUM = [
    "Number of Certified Beds",
    "Average Number of Residents per Day",
    "chain_size",
    "Reported Nurse Aide Staffing Hours per Resident per Day",
    "Reported LPN Staffing Hours per Resident per Day",
    "Reported RN Staffing Hours per Resident per Day",
    "Reported Total Nurse Staffing Hours per Resident per Day",
    "Total number of nurse staff hours per resident per day on the weekend",
    "Registered Nurse hours per resident per day on the weekend",
    "Reported Physical Therapist Staffing Hours per Resident Per Day",
    "Total nursing staff turnover",
    "Registered Nurse turnover",
    "Number of administrators who have left the nursing home",
    "Nursing Case-Mix Index",
]
FS1_CAT = [
    "ownership3", "Urban", "Provider Type", "Provider Resides in Hospital",
    "Continuing Care Retirement Community", "Provider Changed Ownership in Last 12 Months",
    "With a Resident and Family Council", "Automatic Sprinkler Systems in All Required Areas",
    "State", "in_chain",
]

# ---- Feature set 2: FS1 + prior inspection history + engineered features ----
HISTORY = [
    "Rating Cycle 2/3 Total Number of Health Deficiencies",
    "Rating Cycle 2 Number of Standard Health Deficiencies",
    "Rating Cycle 2/3 Number of Complaint Health Deficiencies",
    "Rating Cycle 2/3 Health Deficiency Score",
    "Rating Cycle 2/3 Number of Health Revisits",
    "Rating Cycle 2/3 Health Revisit Score",
    "Rating Cycle 2/3 Total Health Score",
]
ENGINEERED = ["occupancy", "rn_share", "years_certified", "log_beds"]
FS2_NUM = FS1_NUM + HISTORY + ENGINEERED
FS2_CAT = FS1_CAT

FEATURE_SETS = {"FS1": (FS1_NUM, FS1_CAT), "FS2": (FS2_NUM, FS2_CAT)}


def _ownership3(s):
    s = s.fillna("")
    return np.select(
        [s.str.startswith("For profit"), s.str.startswith("Non profit"), s.str.startswith("Government")],
        ["for_profit", "non_profit", "government"], default="other")


def build_features(df):
    """Return (X, y, groups) with all engineered columns added. Drops rows with missing outcome."""
    df = df.loc[df[OUTCOME].notna()].copy()
    in_chain = df["Chain ID"].notna()
    df["in_chain"] = np.where(in_chain, "Y", "N")
    df["chain_size"] = df["Number of Facilities in Chain"].where(in_chain, 1.0)
    df["ownership3"] = _ownership3(df["Ownership Type"])
    beds = df["Number of Certified Beds"].replace(0, np.nan)
    df["occupancy"] = df["Average Number of Residents per Day"] / beds
    df["rn_share"] = (df["Reported RN Staffing Hours per Resident per Day"]
                      / df["Reported Total Nurse Staffing Hours per Resident per Day"].replace(0, np.nan))
    approved = pd.to_datetime(df["Date First Approved to Provide Medicare and Medicaid Services"], errors="coerce")
    df["years_certified"] = (pd.Timestamp(REFERENCE_DATE) - approved).dt.days / 365.25
    df["log_beds"] = np.log1p(df["Number of Certified Beds"])
    # Groups for the clustering check: chain ID, or the facility itself if independent.
    groups = np.where(in_chain, "chain_" + df["Chain ID"].astype("Int64").astype(str), "ind_" + df[ID].astype(str))
    used = set(FS2_NUM + FS2_CAT)
    assert not used & set(LEAKAGE), "leakage column in a feature set"
    return df, df[OUTCOME].astype(float), groups
