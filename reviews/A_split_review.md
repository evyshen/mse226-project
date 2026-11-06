# Review: Person A (Emily): `01_dedup_and_split.py` and write-up draft

Reviewer: Evy (Person C), 2026-11-05. Course references are to MS&E 226 lectures (L#).

**Verdict:** the split code is solid and reproducible, so keep it as is. The blockers are in the write-up and in decisions that haven't been made yet, not in the code.

## What's strong

- Fixed seed (226), the ID column read as text (protects CCNs like `075339` and `25A197`), and a check that dev and holdout share no facility.
- SHA-256 pins on the raw file and on the holdout IDs, plus `split_ccns.csv`, so anyone can confirm or rebuild the exact split. This goes beyond what the handout requires for "save the split".
- Package versions are printed, which handout section 9 asks for.

## Must fix before handoff

1. **The holdout was exposed to an AI tool.** `holdout.csv` was in the Drive folder shared with Claude. Claude did not open it, but the handout says to withhold the holdout from AI tools, and section 4 asks how we did that. Fixes:
    - Move `holdout.csv` to a location only Emily can access.
    - Share only `dev.csv` and `split_ccns.csv`.
    - In section 4, describe the procedure honestly: "the holdout file was never opened or shared with AI tools; one shared folder briefly contained it and the file was not read."
2. **The written sections are still empty.** 1a, 1b, 2, 3a, 3b, 3c, 4 and 6a are headings only. B and C are blocked on **section 2 (outcome)**, because the outcome decides which columns count as leakage (next item) and which metric goes in 6a (L5).
3. **Leakage: decide which columns are available at prediction time (1b, L1, L4).** If the outcome is `Overall Rating`, or a cutoff on it:
    - `Health Inspection Rating`, `Staffing Rating`, `QM Rating` and the long-stay and short-stay QM ratings are **inputs to the Overall Rating formula**. Including them predicts the outcome from its own parts.
    - `Chain Average * Rating` includes the facility's own rating.
    - `High Performing Icon`, `Special Focus Status` and `Abuse Icon` come from the same inspection results.
    - The Health Score, deficiency and staffing-hours columns may be fine, depending on the decision timing stated in 1b.
    - Please write a one-line rule in 1b (for example, "predict before the rating is published, using only inspection and staffing inputs") and a keep/drop list B can apply.
4. **Time columns (handout requirement iv).**
    - Drop `Processing Date` (it is the same value for every row) and both survey dates.
    - `Date First Approved…` can become "years certified" (like house age in L2), but say so in 3a.
5. **Clustering (3a, 7b, L10).** 69% of dev facilities belong to a chain: 611 chains, with the largest at 284 dev rows. Facilities in the same chain share management, so the rows aren't fully independent, and the same chain appears on both sides of the split.
    - Don't redo the split.
    - Do name this in 3a and 7b.
    - For 6b, Evy will compare `KFold` with `GroupKFold(Chain ID)` to see how optimistic the plain CV is.

## Minor

- **3c:** the counts work out to zero duplicates removed (14,686 = 11,748 + 2,938). Report all three checks (exact rows, CCN, name + address) and say none were found. The name + address count is computed but never printed with a decision.
- **Paths:** the script reads `data/NH_ProviderInfo_Oct2026.csv`, but the Drive folder has no raw file and no `data/` folder. Please add the raw file to the team Drive (CMS data is public), so the pinned hash can be checked.
- **Stratification:** not stratifying is fine for a 1–5 rating. If section 2 picks a binary outcome with a rare class, mention it in 4; don't re-split.
- **Missingness:** 24 of 102 columns are more than 30% missing, mostly footnote columns. Mention this in 3a(v) and hand off to B's 5e.
- **Task list:** the doc now assigns 6b and 6c to Evy. Update the "Order of work" bullets, which still say A builds `pipeline.py`.
