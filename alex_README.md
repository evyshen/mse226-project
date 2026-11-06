# MS&E 226 — Alex (Person B) handoff

**Not yet final results.** The provided files contain the original CMS nursing-home CSV, project instructions, and role assignments, but no Emily-created 80% development dataset, agreed outcome, or Evy-created common cross-validation pipeline. The assignment forbids exploring the 20% holdout in Part 1. This script deliberately does not load the original CSV.

## Inputs needed from team
1. Emily: deduplicated 80% development CSV, fixed split seed, and exact outcome definition.
2. Evy: agreed cross-validation folds, seed, and evaluation function (replace this script's provisional 5-fold CV with Evy's common folds before reporting model comparisons).
3. Confirm whether outcome is continuous or binary. `Overall Rating` is only a possible outcome; it is **not** confirmed as the team's choice. If you select rating, consider whether it is a sensible prediction target, and exclude any rating components and other target-leaking variables.

## Run
`python alex_analysis.py --development PATH_TO_EMILY_DEVELOPMENT.csv --outcome 'YOUR OUTCOME COLUMN' --task regression --output alex_results`

For a binary outcome already defined in the development CSV, use `--task classification`.

Outputs: missingness.csv; correlations and plots for continuous outcomes; predictor pair correlations; subgroup tables; two feature sets; and cross-validated linear/logistic model scores. Inspect every selected feature for timing, leakage, appropriateness, and the team's agreed outcome. The automatic feature sets are a **starting proposal**, not final choices.

## Handoff to Emily
Send missingness.csv and a brief recommendation: within each training fold, median-impute numeric features, mode-impute categorical features, and one-hot encode categorical features. Review whether columns with >35% missingness should be excluded and whether missingness is informative. No imputation should be fitted on the holdout or full development data before CV.

## Handoff to Evy
Send the feature_sets object in analysis_metadata.json, the linear_model_cv_results.csv, and any leakage exclusions. Rerun with the exact shared CV folds and metric before combining with Evy's models.

## Suggested report organization
5a: CMS Nursing Home Compare / Care Compare facility-level public reporting. Explain provider self-reporting, administrative sources, inspection-based measures, reporting lags, measurement error, and selection issues. Confirm exact source provenance with Emily.
5b: Name strongest/weakest associations with the **agreed outcome**; include correlation plot and subgroup comparisons; distinguish association from causation.
5c: Report largest absolute covariate pair correlations; discuss multicollinearity and overlapping staffing measures.
5d: Compare ownership type, urban status, chain membership, and provider type if available and substantively appropriate.
5e: Quantify missingness, describe training-fold-only imputation and excluded variables.
6d: Feature set A = operational staffing/size indicators; set B = broader nonleaking facility profile. Explain why operational vs broader predictors might affect generalization. Linear/logistic model provides an interpretable benchmark; include both feature sets and evaluation metrics.

## AI note for Evy (edit to reflect actual use)
Used ChatGPT (GPT-6) to draft a reproducible EDA and regression script and identify potential leakage risks. Checked the script against assignment requirements; pending validation against Emily's development split and Evy's shared evaluation pipeline. Do not claim to have completed checks that were not performed.
