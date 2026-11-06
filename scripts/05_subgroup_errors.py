"""Out-of-fold errors of the final model (lasso, FS2, alpha from 04) by subgroup, for report 7b.
Uses the shared 5 folds; dev data only. Writes results/subgroup_errors.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEV_CSV, ID, RESULTS_DIR               # noqa: E402
from src.features import FEATURE_SETS, build_features         # noqa: E402
from src.pipeline import FOLDS, make_pipeline                  # noqa: E402

if __name__ == "__main__":
    df, y, _ = build_features(pd.read_csv(DEV_CSV, low_memory=False, dtype={ID: str}))
    alpha = json.loads((RESULTS_DIR / "final_model.json").read_text())["alpha"]
    num, cat = FEATURE_SETS["FS2"]
    yhat = cross_val_predict(make_pipeline(Lasso(alpha=alpha, max_iter=50_000), num, cat), df, y, cv=FOLDS)
    err = pd.DataFrame({"y": y.values, "e2": (y.values - yhat) ** 2, "ae": np.abs(y.values - yhat),
                        "ownership": df["ownership3"].values, "chain": df["in_chain"].values,
                        "urban": df["Urban"].values, "hospital": df["Provider Resides in Hospital"].values,
                        "size": pd.qcut(df["Number of Certified Beds"], 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"]).astype(str).values})
    rows = [{"group": "all", "level": "all", "n": len(err), "rmse": np.sqrt(err.e2.mean()), "mae": err.ae.mean(), "mean_y": err.y.mean()}]
    for g in ["ownership", "chain", "urban", "hospital", "size"]:
        for lv, s in err.groupby(g):
            rows.append({"group": g, "level": lv, "n": len(s), "rmse": np.sqrt(s.e2.mean()), "mae": s.ae.mean(), "mean_y": s.y.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "subgroup_errors.csv", index=False)
    print(out.round(2).to_string(index=False))
