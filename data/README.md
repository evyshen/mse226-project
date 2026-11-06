# data/ (not tracked by git)

Put these files here locally. Nothing in this folder is committed.

| File | Who has it | Source |
| --- | --- | --- |
| `NH_ProviderInfo_Oct2026.csv` | everyone | CMS Provider Data Catalog, "Nursing Homes Including Rehab Services: Provider Information", processing date 2026-10-01 (https://data.cms.gov/provider-data/dataset/4pq5-n9py). Must be the October 2026 file: `01_dedup_and_split.py` checks its SHA-256. |
| `dev.csv` | everyone | output of `01_dedup_and_split.py` (11,748 rows) |
| `split_ccns.csv` | everyone | output of `01_dedup_and_split.py` (facility ID -> split) |
| `holdout.csv` | **Person A only, until Part 2** | output of `01_dedup_and_split.py` (2,938 rows). Do not open, plot, or upload to any AI tool. |
