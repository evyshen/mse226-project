# Feature-set suggestions from the EDA (Alex, for discussion with Evy)

Nothing in `src/` was changed. Numbers come from `results/eda/` (written by `scripts/06_eda.py`).
Any change means rerunning `scripts/02`–`05`.

1. **Staffing Rating, or case-mix-adjusted staffing, instead of (or alongside) reported hours.**
   Staffing Rating has r = −0.25 with the outcome, and adjusted total nurse hours have −0.20. Both are stronger
   than any reported staffing column (total −0.14, RN −0.15). Staffing Rating's largest |r| with any FS2 column is
   only 0.57, so P5's "redundant" reason looks weaker than assumed. It isn't leakage (it's not built from the
   inspection). (`candidate_features.csv`)
2. **QM ratings: the EDA suggests low priority.** Emily's rule allows them, but their |r| with the outcome is small:
   QM −0.07, long-stay 0.01, short-stay −0.13 (19.6% missing), chain-average QM 0.00. If we add them, the overall
   QM Rating (1.1% missing) is the cleanest. Chain-average staffing (−0.20) is blank for non-chain facilities, so it
   would need explicit coding like `chain_size`.
3. **Resident and Family Council.** The levels are only "Both", "Resident" and blank (5.1%). Blank probably means
   "no council", but the mode imputation turns it into "Resident". Consider coding blank as its own level. (Check
   the CMS data dictionary first.) (`categorical_outcome.csv`)
4. **Near-duplicates.** Prior deficiency score vs prior total health score, r = 0.98. Weekday vs weekend nurse hours,
   0.98. RN vs weekend RN hours, 0.95. Lasso handles these, but dropping one of each pair would make the OLS
   coefficients easier to read. (`top_feature_pairs.csv`)
5. **Small states.** GU (1 facility), PR (6), DC (13) and AK (16) give state dummies fitted on very few rows. That's
   fine under lasso, but worth one sentence in 7b. (`state_means.csv`)
6. **Weak features.** Occupancy (r = 0.01), LPN hours (0.05), PT hours (−0.06) and chain size (0.07) are near zero
   linearly. Keep them, since RF/lasso can ignore them, but don't over-interpret their coefficients.
