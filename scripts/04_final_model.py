"""Section 6f: refit the selected combination (lasso, FS2) on ALL development data and save it.

alpha is re-chosen on the full dev data with the shared 5 folds and the same grid used in CV.
The saved object is one sklearn Pipeline (imputation + encoding + scaling + lasso), so Part 2
only needs build_features() and pipeline.predict(). Writes models/final_pipeline.joblib
(committed; saved with scikit-learn 1.7.2, load it with the same version) and results/final_model.json + results/final_coefficients.csv.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEV_CSV, ID, MODELS_DIR, OUTCOME, RESULTS_DIR, SEED  # noqa: E402
from src.features import FEATURE_SETS, build_features                     # noqa: E402
from src.pipeline import FOLDS, make_pipeline                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                        # noqa: E402
LASSO_ALPHAS = import_module("02_cv_models").LASSO_ALPHAS

if __name__ == "__main__":
    df, y, _ = build_features(pd.read_csv(DEV_CSV, low_memory=False, dtype={ID: str}))
    num, cat = FEATURE_SETS["FS2"]
    search = GridSearchCV(make_pipeline(Lasso(max_iter=50_000, random_state=SEED), num, cat),
                          {"model__alpha": LASSO_ALPHAS}, cv=FOLDS, scoring="neg_root_mean_squared_error")
    search.fit(df, y)        # refit=True: best alpha refit on all 11,717 dev rows
    final = search.best_estimator_
    alpha = search.best_params_["model__alpha"]

    names = final.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(final.named_steps["model"].coef_, index=names)
    nz = coef[coef != 0].sort_values(key=np.abs, ascending=False)
    nz.rename("coef_per_1sd_or_vs_baseline_level").to_csv(RESULTS_DIR / "final_coefficients.csv")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODELS_DIR / "final_pipeline.joblib")
    meta = {
        "algorithm": "Lasso", "feature_set": "FS2", "outcome": OUTCOME, "alpha": alpha,
        "cv_rmse_at_alpha_nonnested": -search.best_score_,
        "n_train_rows": len(df), "n_features_after_encoding": len(coef), "n_nonzero": int((coef != 0).sum()),
        "intercept": float(final.named_steps["model"].intercept_),
        "threshold": None, "seed": SEED, "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (RESULTS_DIR / "final_model.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(nz.head(15).round(3).to_string())
