"""Exploratory data analysis for report Section 5 (Person B, Alex). Dev data only.

Loads data/dev.csv through src.features.build_features(), so the outcome, the 31 dropped rows
(missing outcome), engineered columns and feature names match the modeling scripts exactly.
Pearson correlations only (df.corr()); no hypothesis tests. Runs in well under 2 minutes.

Writes CSVs to results/eda/ and figures to figures/:
  results/eda/dataset_facts.csv          rows, dropped rows, chain share, inspection date ranges (5a)
  results/eda/outcome_summary.csv        n, mean, median, SD, min, max, skewness, share of zeros
  results/eda/corr_with_outcome.csv      Pearson r of every numeric FS2 feature with the outcome (5b)
  results/eda/categorical_outcome.csv    outcome mean/median/n by level of each categorical FS2 feature (5b)
  results/eda/state_means.csv            outcome mean/median/n by state (5b, Figure 1d)
  results/eda/top_feature_pairs.csv      top 10 |r| pairs among numeric FS2 features (5c)
  results/eda/feature_corr_matrix.csv    full Pearson matrix of numeric FS2 features (5c)
  results/eda/crosstab_ownership_chain.csv   ownership x chain membership, counts and row shares (5c)
  results/eda/subgroup_means.csv         means of key covariates and the outcome by subgroup (5d)
  results/eda/missingness_fs2.csv        missing share of every FS2 feature and the outcome (5e)
  results/eda/missing_over30_by_type.csv count of full-file columns >30% missing, by type (5e)
  results/eda/missing_over30_columns.csv the columns behind that count (5e)
  results/eda/candidate_features.csv     r and missingness of non-leakage columns NOT in FS2 (feature-set suggestions)
  figures/fig1.pdf (+ fig1.png)          4-panel figure for the report's figures page
  figures/boxplots_chain_urban.pdf       outcome by chain membership and by urban/rural
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEV_CSV, ID, OUTCOME, RESULTS_DIR      # noqa: E402
from src.features import FEATURE_SETS, LEAKAGE, build_features  # noqa: E402

EDA_DIR = RESULTS_DIR / "eda"
FIG_DIR = ROOT / "figures"
NUM, CAT = FEATURE_SETS["FS2"]
PRIOR = "Rating Cycle 2/3 Total Number of Health Deficiencies"

# Short labels for figures only (CSV files keep the full CMS column names).
SHORT = {
    "Rating Cycle 2/3 Total Number of Health Deficiencies": "Prior deficiencies",
    "Rating Cycle 2 Number of Standard Health Deficiencies": "Prior standard def.",
    "Rating Cycle 2/3 Number of Complaint Health Deficiencies": "Prior complaint def.",
    "Rating Cycle 2/3 Health Deficiency Score": "Prior deficiency score",
    "Rating Cycle 2/3 Number of Health Revisits": "Prior revisits",
    "Rating Cycle 2/3 Health Revisit Score": "Prior revisit score",
    "Rating Cycle 2/3 Total Health Score": "Prior total score",
    "Number of Certified Beds": "Certified beds",
    "Average Number of Residents per Day": "Residents per day",
    "chain_size": "Chain size",
    "Reported Nurse Aide Staffing Hours per Resident per Day": "Aide hrs/res-day",
    "Reported LPN Staffing Hours per Resident per Day": "LPN hrs/res-day",
    "Reported RN Staffing Hours per Resident per Day": "RN hrs/res-day",
    "Reported Total Nurse Staffing Hours per Resident per Day": "Total nurse hrs/res-day",
    "Total number of nurse staff hours per resident per day on the weekend": "Weekend nurse hrs/res-day",
    "Registered Nurse hours per resident per day on the weekend": "Weekend RN hrs/res-day",
    "Reported Physical Therapist Staffing Hours per Resident Per Day": "PT hrs/res-day",
    "Total nursing staff turnover": "Nursing turnover (%)",
    "Registered Nurse turnover": "RN turnover (%)",
    "Number of administrators who have left the nursing home": "Administrators left",
    "Nursing Case-Mix Index": "Case-mix index",
    "occupancy": "Occupancy", "rn_share": "RN share of hours",
    "years_certified": "Years certified", "log_beds": "log(1+beds)",
}
OWN_LABEL = {"for_profit": "For-profit", "non_profit": "Non-profit", "government": "Government"}

# Key covariates for the 5d subgroup comparison.
KEY = [
    "Reported Total Nurse Staffing Hours per Resident per Day",
    "Reported RN Staffing Hours per Resident per Day",
    "Total nursing staff turnover",
    "Number of Certified Beds",
    "Nursing Case-Mix Index",
    PRIOR,
    OUTCOME,
]


def save(df, name, index=False):
    df.to_csv(EDA_DIR / name, index=index)
    return df


def main():
    t0 = time.time()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DEV_CSV, low_memory=False, dtype={ID: str})   # same call as scripts/05
    df, y, groups = build_features(raw)
    pd.set_option("display.width", 200)

    # ---------------- 5a: dataset facts used in the data-collection paragraph ----------------
    in_chain = df["in_chain"] == "Y"
    d1 = pd.to_datetime(df["Rating Cycle 1 Standard Survey Health Date"], errors="coerce")  # described, never a feature
    d2 = pd.to_datetime(df["Rating Cycle 2 Standard Health Survey Date"], errors="coerce")
    facts = [
        ("rows_in_dev_csv", len(raw)),
        ("rows_missing_outcome_dropped", len(raw) - len(df)),
        ("rows_analyzed", len(df)),
        ("columns_in_file", raw.shape[1]),
        ("n_FS2_numeric_features", len(NUM)),
        ("n_FS2_categorical_features", len(CAT)),
        ("share_in_chain", in_chain.mean()),
        ("n_chains", df.loc[in_chain, "Chain ID"].nunique()),
        ("largest_chain_rows_in_dev", df.loc[in_chain, "Chain ID"].value_counts().max()),
        ("n_states", df["State"].nunique()),
        ("cycle1_survey_date_min", d1.min().date()),
        ("cycle1_survey_date_max", d1.max().date()),
        ("cycle1_share_before_2023", (d1 < "2023-01-01").mean()),
        ("cycle2_survey_date_min", d2.min().date()),
        ("cycle2_survey_date_max", d2.max().date()),
        ("processing_date", raw["Processing Date"].dropna().unique()[0]),
        # 5e: what complete-case analysis (dropping rows with any missing FS2 feature) would lose
        ("rows_with_any_missing_FS2", int(df[NUM + CAT].isna().any(axis=1).sum())),
        ("share_rows_with_any_missing_FS2", df[NUM + CAT].isna().any(axis=1).mean()),
        ("outcome_mean_complete_rows", y[~df[NUM + CAT].isna().any(axis=1)].mean()),
        ("outcome_mean_incomplete_rows", y[df[NUM + CAT].isna().any(axis=1)].mean()),
        ("share_for_profit_complete_rows", (df.loc[~df[NUM + CAT].isna().any(axis=1), "ownership3"] == "for_profit").mean()),
        ("share_for_profit_incomplete_rows", (df.loc[df[NUM + CAT].isna().any(axis=1), "ownership3"] == "for_profit").mean()),
    ]
    facts = save(pd.DataFrame(facts, columns=["fact", "value"]), "dataset_facts.csv")

    # ---------------- 5b(i): outcome summary ----------------
    summ = pd.DataFrame([{
        "n": int(y.size), "mean": y.mean(), "median": y.median(), "sd": y.std(),
        "min": y.min(), "max": y.max(), "skewness": y.skew(), "share_zero": (y == 0).mean(),
        "p90": y.quantile(0.9), "p99": y.quantile(0.99),
    }])
    save(summ, "outcome_summary.csv")

    # ---------------- 5b(ii): Pearson r of every numeric FS2 feature with the outcome ----------------
    r = df[NUM].corrwith(y)                               # Pearson, pairwise complete
    corr_y = pd.DataFrame({"feature": r.index, "r": r.values, "abs_r": r.abs().values,
                           "n_nonmissing": df[NUM].notna().sum().values,
                           "group": ["history" if c.startswith("Rating Cycle") else
                                     "engineered" if c in ("occupancy", "rn_share", "years_certified", "log_beds") else
                                     "profile/staffing" for c in r.index]})
    corr_y = save(corr_y.sort_values("abs_r", ascending=False).reset_index(drop=True), "corr_with_outcome.csv")

    # ---------------- 5b(iii): outcome by level of each categorical FS2 feature ----------------
    rows = []
    for c in CAT:
        g = y.groupby(df[c].fillna("(missing)"))
        for lv, s in g:
            rows.append({"feature": c, "level": lv, "n": s.size, "mean": s.mean(), "median": s.median(), "sd": s.std()})
    cat_tab = save(pd.DataFrame(rows), "categorical_outcome.csv")
    state = (cat_tab[cat_tab.feature == "State"].drop(columns="feature")
             .rename(columns={"level": "state"}).sort_values("mean", ascending=False).reset_index(drop=True))
    save(state, "state_means.csv")

    # ---------------- 5c: correlations among numeric FS2 features ----------------
    cm = df[NUM].corr()
    save(cm, "feature_corr_matrix.csv", index=True)
    iu = np.triu_indices(len(NUM), k=1)
    pairs = pd.DataFrame({"feature_1": cm.index[iu[0]], "feature_2": cm.columns[iu[1]], "r": cm.values[iu]})
    pairs["abs_r"] = pairs.r.abs()
    top = save(pairs.sort_values("abs_r", ascending=False).head(10).reset_index(drop=True), "top_feature_pairs.csv")

    ct = pd.crosstab(df["ownership3"], df["in_chain"], margins=True, margins_name="total")
    ct.columns = [f"n_chain_{c}" if c != "total" else "n_total" for c in ct.columns]
    ct["share_in_chain"] = ct["n_chain_Y"] / ct["n_total"]
    save(ct.reset_index().rename(columns={"ownership3": "ownership"}), "crosstab_ownership_chain.csv")

    # ---------------- 5d: subgroup means of key covariates ----------------
    sub = df[KEY].copy()
    rows = []
    for gname, col in [("ownership", "ownership3"), ("chain", "in_chain"), ("urban", "Urban")]:
        for lv, s in sub.groupby(df[col]):
            row = {"group": gname, "level": lv, "n": len(s)}
            row.update({k: s[k].mean() for k in KEY})
            rows.append(row)
    row = {"group": "all", "level": "all", "n": len(sub)}
    row.update({k: sub[k].mean() for k in KEY})
    subg = save(pd.DataFrame([row] + rows), "subgroup_means.csv")

    # ---------------- 5e: missingness ----------------
    miss = pd.DataFrame({
        "column": NUM + CAT + [OUTCOME],
        "kind": ["numeric"] * len(NUM) + ["categorical"] * len(CAT) + ["outcome"],
    })
    miss["share_missing_analyzed_rows"] = [df[c].isna().mean() for c in miss.column]       # what the models see
    miss["share_missing_dev_file"] = [raw[c].isna().mean() if c in raw else np.nan for c in miss.column]
    miss["n_missing_analyzed_rows"] = [int(df[c].isna().sum()) for c in miss.column]
    miss = save(miss.sort_values("share_missing_analyzed_rows", ascending=False).reset_index(drop=True),
                "missingness_fs2.csv")

    rawmiss = raw.isna().mean()
    over = rawmiss[rawmiss > 0.30]

    def kind(c):
        if "footnote" in c.lower():
            return "footnote code"
        if c.startswith("Chain") or c == "Number of Facilities in Chain":
            return "chain affiliation (blank for non-chain)"
        if c in ("Special Focus Status", "Number of Citations from Infection Control Inspections"):
            return "inspection-derived"
        return "other"
    over_df = pd.DataFrame({"column": over.index, "share_missing": over.values,
                            "type": [kind(c) for c in over.index],
                            "in_LEAKAGE": [c in LEAKAGE for c in over.index],
                            "in_FS2": [c in set(NUM + CAT) for c in over.index]})
    save(over_df.sort_values(["type", "share_missing"]), "missing_over30_columns.csv")
    by_type = over_df.groupby("type").size().rename("n_columns").reset_index()
    by_type = pd.concat([by_type, pd.DataFrame([{"type": "total", "n_columns": len(over_df)}])])
    save(by_type, "missing_over30_by_type.csv")

    # ---------------- candidate features for Task 3 (not in FS2, not leakage) ----------------
    cand = ["QM Rating", "Long-Stay QM Rating", "Short-Stay QM Rating", "Staffing Rating",
            "Chain Average QM Rating", "Chain Average Staffing Rating",
            "Reported Licensed Staffing Hours per Resident per Day", "Nursing Case-Mix Index Ratio",
            "Case-Mix Total Nurse Staffing Hours per Resident per Day",
            "Adjusted Total Nurse Staffing Hours per Resident per Day",
            "Adjusted RN Staffing Hours per Resident per Day"]
    assert not set(cand) & set(LEAKAGE)
    cdf = pd.DataFrame({"column": cand,
                        "r_with_outcome": [df[c].corr(y) for c in cand],
                        "share_missing": [df[c].isna().mean() for c in cand],
                        "max_abs_r_with_FS2_numeric": [df[NUM].corrwith(df[c]).abs().max() for c in cand],
                        "most_correlated_FS2_feature": [df[NUM].corrwith(df[c]).abs().idxmax() for c in cand]})
    cdf = save(cdf.sort_values("r_with_outcome", key=np.abs, ascending=False), "candidate_features.csv")

    # ---------------- Figure 1: four panels ----------------
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], width_ratios=[1, 1.15, 1.05], hspace=0.6, wspace=0.95)
    blue, red, grey = "#3b6ea5", "#c0504d", "#7f7f7f"

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(y, bins=np.arange(0, y.max() + 2, 2), color=blue, edgecolor="white", linewidth=0.3)
    ax.axvline(y.mean(), color=red, lw=1, label=f"mean {y.mean():.1f}")
    ax.axvline(y.median(), color="k", lw=1, ls="--", label=f"median {y.median():.0f}")
    ax.set(title="(a) Outcome distribution", xlabel="Health deficiencies\n(most recent rating cycle)",
           ylabel="Facilities")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    t10 = corr_y.head(10).iloc[::-1]
    ax.barh([SHORT.get(f, f) for f in t10.feature], t10.r, color=[blue if v > 0 else red for v in t10.r])
    ax.axvline(0, color="k", lw=0.5)
    ax.set(title="(b) Top 10 |r| with outcome", xlabel="Pearson r with deficiencies")
    ax.tick_params(axis="y", labelsize=7)
    for i, v in enumerate(t10.r):
        ax.text(v + (0.01 if v > 0 else -0.01), i, f"{v:.2f}", va="center", ha="left" if v > 0 else "right", fontsize=6.5)
    ax.set_xlim(min(0, t10.r.min()) - 0.12, t10.r.max() + 0.12)

    ax = fig.add_subplot(gs[0, 2])
    levels = ["for_profit", "non_profit", "government"]
    data = [y[df["ownership3"] == lv] for lv in levels]
    ax.boxplot(data, widths=0.6, showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3),
               medianprops=dict(color=red), showmeans=True,
               meanprops=dict(marker="D", markersize=3, markerfacecolor="k", markeredgecolor="k"))
    ax.set_xticks(range(1, 4), [f"{OWN_LABEL[lv]}\n(n={len(d):,})" for lv, d in zip(levels, data)], fontsize=6.5, rotation=30)
    ax.set(title="(c) Outcome by ownership", ylabel="Health deficiencies")

    ax = fig.add_subplot(gs[1, :])
    small = state["n"] < 20
    ax.bar(state.state, state["mean"], color=np.where(small, "#a9c1dc", blue), width=0.75)
    ax.axhline(y.mean(), color=red, lw=1, ls="--", label=f"all facilities {y.mean():.1f}")
    ax.set(title="(d) Mean health deficiencies by state (sorted)", ylabel="Mean deficiencies", xlim=(-0.7, len(state) - 0.3))
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    from matplotlib.patches import Patch
    h, _ = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color="#a9c1dc")], [f"all facilities {y.mean():.1f}", "state with < 20 facilities"],
              frameon=False, fontsize=7)

    fig.savefig(FIG_DIR / "fig1.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Chain and urban/rural boxplots (5b), kept out of Figure 1 to fit the page.
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
    for ax, col, lv, lab, title in [
        (axes[0], "in_chain", ["Y", "N"], ["Chain", "Independent"], "Outcome by chain membership"),
        (axes[1], "Urban", ["Y", "N"], ["Urban", "Rural"], "Outcome by urban/rural"),
    ]:
        data = [y[df[col] == v] for v in lv]
        ax.boxplot(data, widths=0.6, flierprops=dict(marker=".", markersize=2, alpha=0.3),
                   medianprops=dict(color=red), showmeans=True,
                   meanprops=dict(marker="D", markersize=3, markerfacecolor="k", markeredgecolor="k"))
        ax.set_xticks([1, 2], [f"{l}\n(n={len(d):,})" for l, d in zip(lab, data)])
        ax.set(title=title, ylabel="Health deficiencies")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots_chain_urban.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "boxplots_chain_urban.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- console summary ----------------
    fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3f}")  # noqa: E731
    print("== dataset facts ==\n" + fmt(facts))
    print("\n== outcome summary ==\n" + fmt(summ))
    print("\n== Pearson r with outcome (FS2 numeric) ==\n" + fmt(corr_y[["feature", "r", "n_nonmissing"]]))
    print("\n== top 10 feature pairs ==\n" + fmt(top))
    print(f"\nDone in {time.time() - t0:.1f}s. CSVs in {EDA_DIR.relative_to(ROOT)}, figures in {FIG_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
