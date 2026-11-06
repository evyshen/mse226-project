"""Sections 6c-6e: evaluate every algorithm x feature set on the shared 5 folds (dev data only).

Tuned models use nested CV: GridSearchCV over 3 inner folds runs inside each outer
training fold, so the outer-fold scores never see data used to pick hyperparameters.

Usage:  python scripts/02_cv_models.py MODEL FS     (MODEL in baseline/ols/lasso/rf, FS in FS1/FS2)
        python scripts/02_cv_models.py all
Writes: results/raw/MODEL_FS.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEV_CSV, ID, RESULTS_DIR, SEED            # noqa: E402
from src.features import FEATURE_SETS, build_features             # noqa: E402
from src.pipeline import FOLDS, INNER_FOLDS, evaluate, make_pipeline  # noqa: E402

LASSO_ALPHAS = np.logspace(-4, 0, 30).tolist()   # sklearn alpha = lambda / (2n)
RF_GRID = {
    "model__max_depth": [None, 15],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": ["sqrt", 0.33],
}
RF_TREES = 300


def spec(model, fs):
    """MODEL may end in '_log': fit on log(1+Y), back-transform with expm1, score on the count scale."""
    if model.endswith("_log"):
        from sklearn.compose import TransformedTargetRegressor
        pipe, grid = spec(model[:-4], fs)
        ttr = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        return ttr, ({f"regressor__{k}": v for k, v in grid.items()} if grid else None)
    num, cat = FEATURE_SETS[fs]
    if model == "baseline":
        return make_pipeline(DummyRegressor(strategy="mean"), num, cat), None
    if model == "ols":
        return make_pipeline(LinearRegression(), num, cat), None
    if model == "lasso":
        pipe = make_pipeline(Lasso(max_iter=50_000, random_state=SEED), num, cat)
        return pipe, {"model__alpha": LASSO_ALPHAS}
    if model == "rf":
        pipe = make_pipeline(RandomForestRegressor(n_estimators=RF_TREES, random_state=SEED, n_jobs=-1),
                             num, cat, scale=False)
        return pipe, RF_GRID
    raise ValueError(model)


def run(model, fs, df, y):
    """Outer-fold loop written out (instead of cross_validate) so long runs can resume:
    each finished outer fold is cached in results/raw/folds/ and skipped on rerun."""
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from scipy.stats import spearmanr
    pipe, grid = spec(model, fs)
    est = pipe if grid is None else GridSearchCV(pipe, grid, cv=INNER_FOLDS,
                                                 scoring="neg_root_mean_squared_error", n_jobs=1)
    cache = RESULTS_DIR / "raw" / "folds"
    cache.mkdir(parents=True, exist_ok=True)
    folds = []
    for k, (tr, va) in enumerate(FOLDS.split(df)):
        f = cache / f"{model}_{fs}_fold{k}.json"
        if not f.exists():
            t = time.time()
            e = clone(est).fit(df.iloc[tr], y.iloc[tr])
            yhat = e.predict(df.iloc[va])
            yv = y.iloc[va]
            f.write_text(json.dumps({
                "rmse": root_mean_squared_error(yv, yhat), "mae": mean_absolute_error(yv, yhat),
                "spearman": spearmanr(yv, yhat).statistic,
                "chosen": {kk: str(v) for kk, v in e.best_params_.items()} if grid else None,
                "seconds": round(time.time() - t, 1)}))
            if time.time() - START > BUDGET:
                print("time budget reached; rerun to resume"); return
        folds.append(json.loads(f.read_text()))
    out = {
        "model": model, "fs": fs,
        "rmse": [d["rmse"] for d in folds], "mae": [d["mae"] for d in folds],
        "spearman": [d["spearman"] for d in folds],
        "grid": {k: [str(v) for v in vs] for k, vs in grid.items()} if grid else None,
        "n_configs": int(np.prod([len(v) for v in grid.values()])) if grid else 0,
        "chosen_per_fold": [d["chosen"] for d in folds] if grid else None,
        "seconds": round(sum(d["seconds"] for d in folds), 1),
    }
    (RESULTS_DIR / "raw" / f"{model}_{fs}.json").write_text(json.dumps(out, indent=2))
    print(f"{model:8s} {fs}: RMSE {np.mean(out['rmse']):.3f} +/- {np.std(out['rmse'], ddof=1):.3f}"
          f"  ({out['seconds']}s)")


START = time.time()
BUDGET = float(__import__("os").environ.get("TIME_BUDGET_S", "1e9"))


if __name__ == "__main__":
    df, y, _ = build_features(pd.read_csv(DEV_CSV, low_memory=False, dtype={ID: str}))
    jobs = ([(m, f) for m in ["baseline", "ols", "lasso", "rf"] for f in ["FS1", "FS2"]]
            + [(m, "FS2") for m in ["ols_log", "lasso_log", "rf_log"]]
            if sys.argv[1] == "all" else [(sys.argv[1], sys.argv[2])])
    for m, f in jobs:
        run(m, f, df, y)
