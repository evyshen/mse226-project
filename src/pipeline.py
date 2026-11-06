"""Shared evaluation code (report Section 6b). Every model must use these folds and evaluate()."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import N_FOLDS, SEED

# ONE fold object for the whole project: identical splits for every comparison.
FOLDS = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
INNER_FOLDS = KFold(n_splits=3, shuffle=True, random_state=SEED)  # tuning inside each outer fold


def group_folds():
    """Chain-grouped folds: no chain appears in both training and validation."""
    return GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def make_preprocessor(num_cols, cat_cols, scale=True):
    """Fit inside each training fold (it sits inside the model Pipeline)."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline(cat_steps), cat_cols),
    ])


def make_pipeline(model, num_cols, cat_cols, scale=True):
    return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, scale)), ("model", model)])


def _spearman(y, yhat):
    return spearmanr(y, yhat).statistic


SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "spearman": make_scorer(_spearman),
}


def evaluate(estimator, X, y, cv=FOLDS, groups=None, n_jobs=1):
    """Return per-fold RMSE, MAE, Spearman (validation folds only) plus fitted fold estimators."""
    out = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=SCORING,
                         return_estimator=True, n_jobs=n_jobs)
    return {
        "rmse": -out["test_rmse"], "mae": -out["test_mae"], "spearman": out["test_spearman"],
        "estimators": out["estimator"],
    }
