# Review: Person B (Alex): `alex_analysis.py` (commit ca2c52b)

Reviewer: Evy (Person C), 2026-11-06. I ran the script on `data/dev.csv` with our outcome
(`Rating Cycle 1 Total Number of Health Deficiencies`, `--task regression`). Lecture numbers refer to the MS&E 226 slides.

**Verdict:** the structure is good: preprocessing is fit inside the folds, it never touches the holdout, it has separate
regression and classification branches, and it watches for leakage. But two bugs make its current output unusable, and it
duplicates code we already share. Fix the two bugs, then build 5a–5e on top of the shared `src/` code.

## What's good

- Imputation, scaling and one-hot encoding sit inside a `Pipeline`, so they're fit only on training folds (L4, handout 6b).
- It refuses to touch the raw file or the holdout.
- It's aware of leakage and handles binary outcomes (StratifiedKFold, accuracy/AUC).
- The README handoff notes and the warning not to over-claim AI checks are good practice for Section 8.

## Must fix

1. **Leakage for our outcome: CV R² = 0.96.** The rating/inspection exclusions only run when `--outcome` is exactly
   `'Overall Rating'`. With our outcome, feature set B keeps 27 leaking columns: every other Rating Cycle 1 column,
   `Total Weighted Health Survey Score`, the health-inspection and overall ratings, and fines and penalties. Result:
   RMSE 1.57 and R² 0.96. That isn't a good model, it's the outcome predicting itself. Use the shared
   `LEAKAGE` list in `src/features.py`, which follows Emily's rule in report Section 1(b).
2. **The substring filter `'id'` drops real predictors.** `'id' in c.lower()` matches "res**id**ent" and "prov**id**er", so the
   script silently drops **every staffing-hours column** ("…per Resident per Day"), Average Number of Residents, Provider Type,
   Provider Resides in Hospital, With a Resident and Family Council, and Provider Changed Ownership. That's why
   feature set A has only 5 columns and no staffing at all (CV R² 0.10). Match whole column names, or use the
   explicit lists in `src/features.py`.
3. **`State` is dropped by the 25-level cardinality cap.** State is one of the strongest predictors, because inspector
   strictness differs by state. Keep it; 53 dummies is fine with n = 11,717.
4. **The README upload replaced the team README.** I restored the team README and moved yours to `alex_README.md`. When you
   upload through the GitHub website, check that it isn't overwriting an existing file with the same name.

## Should fix

- **Use the shared folds and metrics.** The script makes its own `KFold(random_state=42)` and reports RMSE/R² with
  `np.std(ddof=0)`. Model comparisons must use the same splits (handout 6b), so import `FOLDS` and `evaluate` from
  `src/pipeline.py` (seed 226, RMSE + MAE). Model 1 (OLS) has already been run that way: it's in `results/cv_results.csv`,
  and you can take over its write-up.
- **Spearman is everywhere.** We dropped it (not covered in lectures). For 5b/5c use ordinary (Pearson) correlations
  (`df.corr()`), which match the handout's "most strongly correlated" wording. Preliminary Pearson correlations with the outcome:
  prior-cycle deficiencies 0.62, prior health score 0.42, turnover 0.19, beds 0.18, case mix 0.13, staffing −0.14.
- **The feature sets are automatic, not motivated.** "Operational vs broad" is a reasonable idea, but 6d needs a rationale
  tied to the EDA (L2, L7). The report currently uses FS1 (profile + staffing + turnover + state) and FS2 (FS1 + inspection
  history + engineered features). Either adopt those and write the rationale, or propose changes; all models would then be rerun.
- **Feature screening on all of dev** (keeping columns at least 65% non-missing, dropping constant columns) happens before CV.
  It's harmless here, since it doesn't use the outcome, but say so in 5e. Screening that *uses the outcome* would have to go
  inside the folds (L4).
- **Subgroups:** only the first 12 categorical columns get tables, and there are no plots. The handout asks for
  "appropriate plots" in 5b and 5d. Suggest boxplots of the outcome by ownership (3 levels), chain vs independent, urban/rural,
  and a state-mean bar chart.
- **The README is out of date.** dev.csv, the outcome and the shared CV now exist. Please update it.
- **AI note:** check the exact model name and version you used ("GPT-6"?) and what you actually verified, for Section 8.

## What's still needed from Alex for the report

5a (data collection), 5b–5d (correlations, plots, subgroups), 5e (missing-data plan: drop footnotes, median + indicator,
mode; all fit in-fold, as already implemented), the OLS bullet in 6d, and your Section 8 entry. There's room for
about one page of text plus figures.
