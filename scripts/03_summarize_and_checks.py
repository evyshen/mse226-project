"""Build the Section 6e table from results/raw/ and run two robustness checks (dev data only).

1. Clustering check: refit fixed-hyperparameter versions of each FS2 model on
   chain-grouped folds and compare with the ordinary folds (how optimistic is plain CV?).
2. RF grid-edge check: the RF tuner picked the smallest max_features and min_samples_leaf
   in the grid, so we try values beyond the edge on the shared folds.
Writes results/cv_results.csv, results/cv_table.tex, results/group_cv_check.csv,
results/rf_edge_check.csv.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEV_CSV, ID, RESULTS_DIR, SEED                    # noqa: E402
from src.features import FEATURE_SETS, build_features                    # noqa: E402
from src.pipeline import FOLDS, evaluate, group_folds, make_pipeline     # noqa: E402

NAMES = {"baseline": "Baseline (training-fold mean)", "ols": "OLS", "lasso": "Lasso", "rf": "Random forest",
         "ols_log": "OLS, log(1+Y)", "lasso_log": "Lasso, log(1+Y)", "rf_log": "Random forest, log(1+Y)"}


def table():
    rows = []
    for m in NAMES:
        for fs in ["FS1", "FS2"]:
            f = RESULTS_DIR / "raw" / f"{m}_{fs}.json"
            if not f.exists():
                continue  # log-target variants are run on FS2 only
            d = json.loads(f.read_text())
            if m == "baseline" and fs == "FS2":
                continue  # baseline ignores features; identical to FS1
            sel = None
            if d["chosen_per_fold"]:
                c = Counter(json.dumps(x, sort_keys=True) for x in d["chosen_per_fold"])
                sel = "; ".join(f"{json.loads(k)} x{v}" for k, v in c.most_common())
            rows.append({
                "algorithm": NAMES[m], "feature_set": "--" if m == "baseline" else fs,
                "rmse_mean": np.mean(d["rmse"]), "rmse_sd": np.std(d["rmse"], ddof=1),
                "mae_mean": np.mean(d["mae"]), "spearman_mean": np.mean(d["spearman"]),
                "n_configs": d["n_configs"], "selected_per_fold": sel,
            })
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    print(t[["algorithm", "feature_set", "rmse_mean", "rmse_sd", "mae_mean", "spearman_mean"]].round(3).to_string())
    return t


def fixed_models():
    num, cat = FEATURE_SETS["FS2"]
    return {
        "OLS": make_pipeline(LinearRegression(), num, cat),
        "Lasso (alpha=0.0033)": make_pipeline(Lasso(alpha=0.0033, max_iter=50_000), num, cat),
        "RF (sqrt, leaf 1, depth None)": make_pipeline(
            RandomForestRegressor(300, max_features="sqrt", min_samples_leaf=1, random_state=SEED, n_jobs=-1),
            num, cat, scale=False),
    }


if __name__ == "__main__":
    table()
    df, y, groups = build_features(pd.read_csv(DEV_CSV, low_memory=False, dtype={ID: str}))
    num, cat = FEATURE_SETS["FS2"]

    if "--group" in sys.argv:
        rows = []
        for name, est in fixed_models().items():
            a = evaluate(est, df, y, cv=FOLDS)["rmse"]
            b = evaluate(est, df, y, cv=group_folds(), groups=groups)["rmse"]
            rows.append({"model_FS2": name, "rmse_kfold": a.mean(), "rmse_group_kfold": b.mean(),
                         "difference": b.mean() - a.mean()})
        g = pd.DataFrame(rows)
        g.to_csv(RESULTS_DIR / "group_cv_check.csv", index=False)
        print(g.round(3).to_string())

    if "--edge" in sys.argv:
        rows = []
        for mf in [0.1, 0.5, 1.0]:
            est = make_pipeline(RandomForestRegressor(300, max_features=mf, min_samples_leaf=1,
                                                      random_state=SEED, n_jobs=-1), num, cat, scale=False)
            r = evaluate(est, df, y, cv=FOLDS)["rmse"]
            rows.append({"model_FS2": f"RF max_features={mf}", "rmse_mean": r.mean(), "rmse_sd": r.std(ddof=1)})
        est = make_pipeline(HistGradientBoostingRegressor(random_state=SEED), num, cat, scale=False)
        r = evaluate(est, df, y, cv=FOLDS)["rmse"]
        rows.append({"model_FS2": "Gradient boosting (defaults, untuned)", "rmse_mean": r.mean(), "rmse_sd": r.std(ddof=1)})
        e = pd.DataFrame(rows)
        e.to_csv(RESULTS_DIR / "rf_edge_check.csv", index=False)
        print(e.round(3).to_string())
