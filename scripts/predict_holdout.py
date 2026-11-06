"""PART 2 ONLY. Do not run during Part 1.

Person A runs this once, on her own machine, after the team freezes the final pipeline:
    python scripts/predict_holdout.py path/to/holdout.csv
It prints only aggregate scores; no rows are printed or saved.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, MODELS_DIR    # noqa: E402
from src.features import build_features  # noqa: E402

if __name__ == "__main__":
    hold, y, _ = build_features(pd.read_csv(sys.argv[1], low_memory=False, dtype={ID: str}))
    yhat = joblib.load(MODELS_DIR / "final_pipeline.joblib").predict(hold)
    print(f"holdout n={len(y)}  RMSE={root_mean_squared_error(y, yhat):.3f}  "
          f"MAE={mean_absolute_error(y, yhat):.3f}")
