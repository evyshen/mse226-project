# Alex: start here

Hi Alex! This guide gets you set up and gives you a prompt to paste into Claude so it can redo your part in line with
what Emily and Evy have already built. It should take about 30 minutes to set up, then Claude does most of the work and
you check it.

---

## Part 1: One-time setup (no command line needed)

1. **Accept the repo invite.** Check your email for a GitHub invite to `evyshen/mse226-project` and accept it.
   (If there's no invite, send Evy your GitHub username.)
2. **Install GitHub Desktop** (https://desktop.github.com) and sign in with your GitHub account.
3. **Download the repo:** in GitHub Desktop, File → Clone Repository → choose `evyshen/mse226-project` → pick a folder
   such as `Documents/mse226-project` → Clone.
4. **Add the data:** get `dev.csv` from Evy (Drive). Put it in the repo's `data/` folder. Git ignores that folder, so the
   data never gets uploaded. **Never** ask for, open, or give anyone (including Claude) `holdout.csv`.
5. **Make your own branch:** in GitHub Desktop, Current Branch → New Branch → name it `alex/eda` → Create.
   Your work stays separate until Evy reviews it.

**Saving and sharing work later (GitHub Desktop):**

- The left panel lists changed files. Make sure **no `.csv` from `data/`** is listed. Results CSVs in `results/` are fine.
- Type a short summary (e.g. "EDA script and figures") → **Commit to alex/eda** → **Push origin**.
- Click **Create Pull Request**. It opens GitHub in your browser; click Create. Evy reviews it and merges.
- **Don't upload files through the GitHub website.** That's what overwrote the team README last time.

---

## Part 2: What's already decided (read before starting)

- **Outcome:** `Rating Cycle 1 Total Number of Health Deficiencies`. This is the number of health deficiencies at a
  facility's most recent inspection cycle. Stakeholder: a state survey agency deciding which nursing homes to inspect
  first. (Report Sections 1–2, written by Emily.)
- **Data:** `data/dev.csv` = 11,748 facilities (the 80% development split, seed 226). 31 rows have no outcome, so
  11,717 are used.
- **Leakage rule:** anything produced by the inspection being predicted is excluded. The list is in `src/features.py`
  (`LEAKAGE`).
- **Feature sets** (`src/features.py`):
    - FS1 = facility profile, staffing, turnover, case mix, state.
    - FS2 = FS1 + prior inspection history + engineered features.
- **Models are done** (`results/cv_results.csv`): baseline, OLS (your Model 1, already run on the shared folds), lasso and
  random forest. The final model is lasso on FS2. Every decision and its reason is in `DECISIONS.md`.
- **Metrics:** RMSE (primary) and MAE. We **dropped Spearman** because it isn't covered in lectures. Use ordinary
  (Pearson) correlations.
- **The report lives on Overleaf**, not in the repo. Your sections are 5a–5e, the OLS bullet in 6d, your Section 8
  entry, and the Figures page.
- **Page budget:** the report is already over the 5-page limit. Section 5 gets about **0.75 page of text**. The Figures
  page is one page of figures only.

---

## Part 3: Prompt to paste into Claude

Open the **Claude desktop app → Cowork**, select your `mse226-project` folder when asked, and paste everything in the
box below.

```
I'm Person B (Alex) on a 3-person MS&E 226 (Stanford, Fundamentals of Data Science) project, Part 1. The repo is in my
selected folder. My teammates already finished the setup and the models. I need you to redo my part so it matches their
work exactly and meets the project handout. Work step by step and show me results before writing report text.

READ FIRST (in this order): README.md, DECISIONS.md, src/config.py, src/features.py, src/pipeline.py,
reviews/B_alex_analysis_review.md (a review of my earlier script and its bugs), results/cv_results.csv,
results/subgroup_errors.csv. Do not modify anything in src/ or scripts/02-05. If you think something there is wrong,
tell me instead.

HARD RULES
- Use ONLY data/dev.csv. Never look for, open, or ask for holdout.csv or the raw CMS file. If you see them, stop and
  tell me.
- Load data with src.features.build_features() so the outcome, dropped rows (31 missing outcome), engineered columns and
  LEAKAGE list match the team's. Use FEATURE_SETS for feature names.
- Use Pearson correlation (df.corr()), not Spearman. No p-values, no hypothesis tests (not part of Part 1).
- Wording: "associated with", never "causes", "drives" or "leads to".
- Every number in the report text must come from a file the script writes. Do not type numbers from memory.
- Don't reuse my old alex_analysis.py logic. It had a leakage bug (R^2 = 0.96) and an 'id' substring filter that dropped
  every staffing column. Write a new script.

TASK 1: write scripts/06_eda.py (runs in under 2 minutes). It writes CSVs to results/eda/ and figures to figures/:
 a) Outcome summary: n, mean, median, SD, min/max, skewness, share of zeros. Histogram of the outcome.
 b) 5b: Pearson correlation of every numeric FS2 feature with the outcome, sorted by |r|. Report the strongest and
    weakest. For each categorical FS2 feature, outcome mean/median/n by category. Boxplots of the outcome by ownership
    (3 levels), chain vs independent, and urban vs rural. Bar chart of mean deficiencies by state.
 c) 5c: correlations among numeric FS2 features. List the top 10 pairs by |r| (expect near-duplicate staffing measures,
    beds vs residents, prior-cycle history columns). Plus one contingency table: ownership x chain membership.
 d) 5d: subgroups (ownership, chain vs independent, urban vs rural). Compare means of key covariates across groups:
    total nurse staffing hours, RN hours, nursing turnover, beds, case-mix index, prior-cycle deficiencies, outcome.
 e) 5e: missingness of every FS2 feature plus the outcome, and the count of columns in the full file over 30% missing
    by type (footnote codes, chain-affiliation fields that are blank for non-chain facilities, inspection-derived).
 f) ONE multi-panel figure for the report's figures page, figures/fig1.pdf: (1) outcome histogram, (2) top-10 |r|
    bar chart, (3) boxplots by ownership, (4) mean deficiencies by state. Clear titles, units ("deficiencies"),
    readable at half-page size.
Run it, then show me the key tables and the figure before going on.

TASK 2: draft LaTeX for Overleaf. I paste it in myself, so output plain LaTeX in a code block. Match the existing style:
\textbf{(a) Data collection.} ... paragraphs, 11pt, no new packages. Budget: about 0.75 page total for 5a-5e.
 - 5a: how CMS collects this data. State survey agencies run standard inspections (federally required within
   15 months) and complaint investigations. Staffing comes from payroll-based (PBJ) reports. Ratings come from the
   five-star methodology. Issues that could affect our analysis: self-reported or administrative data, inspection dates
   spanning 2019-2026 while staffing is recent, differences in inspector strictness by state, chain clustering
   (69% of facilities in 611 chains), and complaint deficiencies depending on complaints being filed. Keep facts to
   what you can support; flag anything you're unsure of with \TODO{}.
 - 5b, 5c: strongest and weakest correlations, anything surprising (e.g. how weak staffing is), categorical
   comparisons, a pointer to Figure 1.
 - 5d: subgroup differences with numbers.
 - 5e: the missing-data plan, which must match src/pipeline.py exactly: median imputation plus missing-indicator
   columns for numeric features, mode for categorical, fit inside each CV training fold; footnote columns dropped;
   chain fields coded explicitly for non-chain facilities; 31 rows with a missing outcome dropped. Say why we impute
   rather than drop (missingness is modest, 2.5-13% for used features, and dropping rows would lose facilities
   non-randomly).
 - The OLS bullet for 6d (2 sentences): how OLS makes predictions, and why it's our interpretable benchmark (Lecture 2;
   Gauss-Markov, Lecture 7). Do not change Evy's lasso/RF bullets.
 - The Figures page: \includegraphics[width=\linewidth]{figures/fig1.pdf} with a one-line caption.
 - Cite lectures where relevant (e.g. "Lecture 4: imputation must be fit inside each training fold").

TASK 3: point out anything in the feature sets that the EDA suggests changing (e.g. adding the QM ratings, which
Emily's rule in Section 1(b) allows). Do NOT change the feature sets yourself. List the suggestions and I'll discuss
them with Evy.

TASK 4: draft my Section 8 (AI tools) entry, 3 sentences: tools and versions used (ask me), what I delegated, how I
checked it, plus one concrete suggestion I accepted, modified or rejected (ask me). Don't claim checks I didn't do.

When finished, list every file you created or changed, so I can commit them on my alex/eda branch in GitHub Desktop.
```

---

## Part 4: Checklist before you open the pull request

- [ ] `python scripts/06_eda.py` runs start to finish on your machine (or in Claude) without errors.
- [ ] No `data/*.csv` in the GitHub Desktop changes list.
- [ ] Every number in your LaTeX text appears in a `results/eda/*.csv` file.
- [ ] You can explain every plot and number in your own words (the oral presentation tests this).
- [ ] Sections 5a–5e, the OLS bullet and your Section 8 entry are pasted into Overleaf; your last name is in the
      author line.
- [ ] `figures/fig1.pdf` is uploaded to Overleaf (Upload → into a `figures` folder) and the report compiles.
