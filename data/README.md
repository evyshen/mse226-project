# data/

| File | In git? | Notes |
| --- | --- | --- |
| `dev.csv` | **yes** | 80% development split (11,748 rows), the only data Part 1 code reads. Output of `01_dedup_and_split.py`. |
| `split_ccns.csv` | **yes** | Facility ID -> `dev`/`holdout` label for all 14,686 facilities, so the split can be rebuilt from IDs alone. IDs only, no outcomes. |
| `NH_ProviderInfo_Oct2026.csv` | **no** | Raw CMS file, processing date 2026-10-01 (https://data.cms.gov/provider-data/dataset/4pq5-n9py). Not committed because it contains the holdout rows. `01_dedup_and_split.py` checks its SHA-256. CMS replaces this file monthly, so a copy is kept on the team Drive. |
| `holdout.csv` | **never** | 20% holdout (2,938 rows). Person A only, until Part 2. Never opened, plotted, committed, or given to an AI tool. |

All of these are public U.S. government data (CMS, not copyrighted; 17 U.S.C. § 105). The raw file and the holdout are
kept out of git to protect the holdout, not because of licensing.
