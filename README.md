# MS&E 226 Project: Nursing home ratings (CMS Provider Information, Oct 2026)

Team: Emily (Person A), Alex (Person B), Evy (Person C).

## Layout

```
01_dedup_and_split.py   Step 1: duplicate checks + seeded 80/20 split (Person A)
src/                    shared code: config.py (outcome, seed), features.py (feature sets,
                        leakage list), pipeline.py (CV folds, in-fold preprocessing, evaluate())
scripts/                02 CV for all models, 03 results table + checks, 04 final model,
                        05 subgroup errors, 06 EDA for Section 5 (Alex), predict_holdout.py (Part 2 only)
figures/                Figure 1 for the report (written by 06_eda.py)
results/                aggregate CV results (no data rows)
DECISIONS.md            every modeling decision with its rationale
notebooks/              EDA and modeling notebooks, one per person/topic
reviews/                code and write-up review notes
data/                   local only, never committed (see data/README.md)
```

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# dev.csv is already in data/. To rebuild the split, put NH_ProviderInfo_Oct2026.csv in data/ and run:
python 01_dedup_and_split.py
python scripts/02_cv_models.py all
python scripts/03_summarize_and_checks.py --group --edge
python scripts/04_final_model.py
python scripts/05_subgroup_errors.py
python scripts/06_eda.py
```

Expected output: 14,686 raw rows, 11,748 dev, 2,938 holdout, seed 226, and a holdout ID fingerprint matching the one in the script.

## Ground rules

1. **Only `data/dev.csv` and `data/split_ccns.csv` are in git.** `.gitignore` blocks every other CSV in `data/`, and the raw file and holdout by name. Check `git status` before every commit.
2. **The holdout is untouched until Part 2.** Only Person A keeps `holdout.csv`. Nobody loads it, and nobody gives it, its folder, or the raw file plus `split_ccns.csv` to an AI tool.
3. **One fold object for every model.** Import the CV folds from `src/`. Don't create your own `KFold`.
4. **Fit preprocessing inside the folds.** Imputation, scaling and feature selection go inside an sklearn `Pipeline`.
5. **Branch, then pull request.** Work on `name/topic` branches and open a PR into `main`. Evy reviews before merging.
6. **Fixed seeds everywhere:** `random_state=226`.
