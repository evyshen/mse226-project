# Decision log: Person C (Evy), Part 1

Every modeling choice, the alternatives considered, and why. Lecture numbers refer to the MS&E 226 slides.
Choices marked **provisional** belong to Emily (outcome, stakeholder) or Alex (feature sets, Model 1). They were set
so the pipeline could run, and they change in one place (`src/config.py`, `src/features.py`).

## Inputs assumed (provisional, owned by A and B)

| # | Decision | Rationale |
|---|---|---|
| P1 | Outcome = `Rating Cycle 1 Total Number of Health Deficiencies` (state survey agency stakeholder) | Avoids recomputing CMS's Overall Rating formula; a count in the agency's units (L1, L5). |
| P2 | Drop 31 dev rows with missing outcome, leaving 11,717 | You can't evaluate a prediction without the outcome. That's 0.3% of rows, so the loss is negligible. |
| P3 | Leakage list in `src/features.py` (`LEAKAGE`), enforced by an assert | Everything from, or built on, the inspection being predicted: Cycle 1 columns, the weighted score, health and overall ratings and their chain averages, icons, fines and penalties, the ">2 years ago" flag (L4: the prediction must use only information available beforehand). |
| P4 | FS1 = facility profile, staffing, turnover, case mix, state (24 columns). FS2 = FS1 + Cycle 2/3 inspection history + 4 engineered features (35 columns) | Tests the stakeholder question "how much does inspection history add?" The engineered features (occupancy, RN share, years certified, log beds) give linear models some nonlinearity (L2). |
| P5 | Excluded QM ratings and Staffing Rating from both sets | The Staffing Rating is computed from the staffing hours already included, so it is redundant. QM ratings are another CMS rating, and leaving them out keeps the story clean. Alex can test adding them. |
| P6 | Ownership collapsed from 13 levels to 3 (for-profit / non-profit / government); dates turned into years certified relative to the fixed processing date 2026-10-01 | Fewer sparse dummy columns, so lower variance (L7). The handout bans raw time features; facility age is like house age (L2). The reference date is a documented constant, not estimated from the data. |

## Section 6b: evaluation procedure

| # | Decision | Alternatives | Rationale |
|---|---|---|---|
| C1 | 5-fold `KFold(shuffle=True, random_state=226)`, created once in `src/pipeline.py` and used by every model | 10-fold; a single validation split | K = 5 to 10 is standard. 5 keeps the nested random forest affordable, and each model still trains on 80% of the data. One fold object means identical splits for every comparison, as the handout requires (L4). |
| C2 | Preprocessing inside the `Pipeline`: median imputation with missing-indicator columns, one-hot encoding (drop first, ignore unseen levels), standardization | Imputing once on all the data | Imputation and scaling are estimated from data, so they must be fit on training folds only (L4, handout 6b). Missing indicators keep the signal that "turnover not reported" may itself be informative. Standardization is required for lasso (L3). |
| C3 | Feature engineering is row-wise only, so it runs before CV | Engineering inside the folds | Ratios, logs and age use only a row's own values plus constants, so no information crosses folds. |
| C4 | Tuning by **nested CV**: GridSearchCV over 3 inner folds inside each outer training fold | Tuning on the same 5 folds that are reported | Tuning on the reported folds gives an optimistic score, the winner's curse (L3, L4). Nested CV keeps the reported scores honest for the tuned models. |
| C5 | Metrics: RMSE primary; MAE and Spearman rank correlation secondary | R², accuracy on binned counts | RMSE is in deficiencies and penalizes big misses, which are the facilities an agency must catch (L2). Spearman measures how well the model **ranks** facilities, which is the agency's actual decision (inspection order). R² is in-sample in spirit (L2). |
| C6 | Clustering check: refit the FS2 models on chain-grouped folds (`GroupKFold`; each independent facility is its own group) | Ignore clustering | 69% of facilities are in chains, so plain CV may leak chain-level information across folds (L10). |

## Section 6c: baseline

| # | Decision | Rationale |
|---|---|---|
| C7 | `DummyRegressor(strategy="mean")`, which recomputes the training-fold mean in each fold | The constant that minimizes squared error is the mean (L2). It answers "what if the agency just assumes the national average?" |

## Section 6d: algorithms and tuning

| # | Decision | Alternatives | Rationale |
|---|---|---|---|
| C8 | Model 2 = **lasso** | Ridge | FS2 has near-duplicate columns (weekday vs weekend staffing; several history scores). Lasso can zero out redundant ones and gives a shorter list of features to explain (L3, L7). Ridge would also be defensible. |
| C9 | Lasso grid: alpha in 30 log-spaced values from 1e-4 to 1 (sklearn alpha = lambda/2n) | A wider grid | The selected alphas (0.0017 to 0.0033) sit well inside the grid, so the optimum is not at an edge (L3). |
| C10 | Model 3 = **random forest** | Gradient boosting / XGBoost | Few hyperparameters that matter, robust under time pressure, no scaling needed (L6). Boosting was checked as a sensitivity (C14). |
| C11 | RF grid: max_depth {None, 15} x min_samples_leaf {1, 5, 20} x max_features {sqrt, 0.33}, giving 12 settings; 300 trees; seed 226 | A larger grid | 12 settings x 3 inner folds x 5 outer folds = 180 fits per feature set is affordable. Depth and leaf size control the bias-variance tradeoff (L7). Tree count isn't tuned because more trees only lowers variance. |
| C12 | OLS (Model 1) run in the same harness so the 6e table is complete | Wait for Alex | Same folds and preprocessing as every other model. Alex owns its write-up and can replace it. |

## Section 6e: results (dev data, 5-fold CV, seed 226)

| Algorithm | FS | RMSE mean +/- SD | MAE | Spearman | Tuning (tried -> selected) |
|---|---|---|---|---|---|
| Baseline (mean) | -- | 8.065 +/- 0.176 | 5.870 | -- | -- |
| OLS | FS1 | 6.649 +/- 0.179 | 4.729 | 0.587 | none |
| OLS | FS2 | 5.912 +/- 0.112 | 4.207 | 0.671 | none |
| Lasso | FS1 | 6.649 +/- 0.179 | 4.727 | 0.586 | 30 alphas -> 0.0013 to 0.0024 |
| **Lasso** | **FS2** | **5.904 +/- 0.104** | **4.199** | **0.671** | 30 alphas -> 0.0017 to 0.0033 |
| Random forest | FS1 | 6.620 +/- 0.149 | 4.754 | 0.566 | 12 settings -> depth None, leaf 1, sqrt (4 of 5 folds) |
| Random forest | FS2 | 6.031 +/- 0.086 | 4.332 | 0.642 | 12 settings -> depth None, leaf 1, sqrt (5 of 5 folds) |

**Reading the table:**

- **Every model beats the baseline by a wide margin.** FS2 models cut RMSE by about 27% (8.07 to 5.90), which explains roughly 46% of the variance out of fold.
- **Adding inspection history (FS1 to FS2) is the biggest gain:** about 0.74 deficiencies for the linear models and 0.59 for RF. That is 4 to 7 times the fold SD, so it's real and matters to the agency. Prior deficiencies alone have a Spearman correlation of about 0.5 with the outcome.
- **Algorithm differences within a feature set are small.** Lasso and OLS are tied: 0.008 apart, with the per-fold differences changing sign. The lasso chose very small alphas, so the variance reduction over OLS is slight; n = 11,717 is large relative to about 105 encoded features, so OLS variance is already low (L7).
- **RF is slightly worse on FS2** (+0.13 RMSE, worse in all 5 folds). The strongest signal (prior deficiencies to current deficiencies) is close to linear, so the forest's flexibility adds variance without reducing bias much. On FS1, RF ties the linear models.
- **Practically meaningful?** For the agency, a gap of 0.1 deficiencies is not meaningful; choosing FS2 (history) is.

## Robustness checks

| # | Check | Result | Decision |
|---|---|---|---|
| C13 | Chain-grouped CV vs plain CV (FS2, fixed hyperparameters) | OLS +0.061, lasso +0.027, RF +0.074 RMSE | Plain CV is slightly optimistic because of chain clustering, but the ranking of models doesn't change. Report this in 6b and 7b. |
| C14 | RF beyond the grid edge (max_features 0.1 / 0.5 / 1.0) and untuned gradient boosting | 6.031 / 6.056 / 6.096; boosting 5.940 +/- 0.054 | Widening the grid doesn't help RF. Untuned boosting comes close to lasso but doesn't beat it. Nonlinear methods don't buy accuracy here. |

## Section 6f: final model

| # | Decision | Rationale |
|---|---|---|
| C15 | **Lasso on FS2**, alpha re-chosen on all dev data with the shared folds (alpha = 0.0024; 84 of 105 coefficients nonzero; non-nested CV RMSE 5.905), refit on all 11,717 dev rows | It has the lowest RMSE on the primary metric. It is tied with OLS, but the shrinkage stabilizes collinear staffing and history columns, and it was the least optimistic model under chain-grouped CV (C13). It is also interpretable for the agency, unlike RF (L6). |
| C16 | Save one sklearn Pipeline (`models/final_pipeline.joblib`, gitignored, regenerated by `scripts/04_final_model.py`) plus `results/final_model.json` | The handout requires saving the fitted preprocessing, feature definitions and tuning choices. There is no classification threshold (continuous outcome). |
| C17 | Part 2 script `scripts/predict_holdout.py` prints only aggregate metrics; Emily runs it once | Keeps the holdout untouched and never displayed (handout Section 4). |

**Interpretation (standardized coefficients, holding the other features fixed; associations, not causes):**

- Prior-cycle deficiencies are the strongest predictor: +5.0 deficiencies per SD.
- Positive associations: larger facilities (+0.9 per SD of log beds), a recent ownership change (+0.8), administrator turnover (+0.5 per SD), RN and nursing turnover (+0.5 each).
- Non-profit (−0.6) and government (−0.5) ownership are associated with fewer deficiencies than for-profit.
- More nurse staffing hours: −0.4 per SD.
- 40 of 52 state dummies are nonzero, with large spreads (e.g. DC +11.4 and AR −4.9 relative to AK). State differences likely reflect inspector strictness as well as care quality, which is a key limitation for 7b.

## Reproduce

```bash
python 01_dedup_and_split.py                      # Emily; needs the raw CMS file
python scripts/02_cv_models.py all                # ~10 min on 4 cores; resumable (cached per fold)
python scripts/03_summarize_and_checks.py --group --edge
python scripts/04_final_model.py
```

Tested with Python 3.10.12, pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3. `GroupKFold(shuffle=...)` needs scikit-learn 1.6 or later.
