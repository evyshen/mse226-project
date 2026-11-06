# Alex: start here

Hi Alex! You **don't need to learn git or GitHub** for this. Claude does the analysis, you check it, and Evy puts it on
GitHub. Plan for about 15 minutes of setup, then Claude does most of the work.

**Who does what:**

| Step | You | Claude | Evy |
| --- | --- | --- | --- |
| Get the files | Download 2 things (below) | | Shares dev.csv |
| Do the analysis | Paste the prompt, answer its questions, check results | Writes code, runs it, makes figures, drafts text | |
| Hand it in | Send Evy one ZIP, paste the text into Overleaf | | Commits your files to GitHub |

> **Why not let Claude connect to GitHub directly?** We tried. The GitHub connector fails to sign in from the Claude app,
> and Claude can't push to GitHub for you anyway. Downloading a ZIP is the simplest reliable path.

---

## Step 1: Download the team's code (2 min)

1. Open **https://github.com/evyshen/mse226-project** in your browser. (You don't need to log in; the repo is public.)
2. Click the green **`<> Code`** button → **Download ZIP**.
3. Open your Downloads folder and **double-click `mse226-project-main.zip`** to unzip it. You'll get a folder called
   `mse226-project-main`.
4. Move that folder somewhere easy to find, e.g. **Documents**. Rename it to `mse226-project` if you like.

## Step 2: Add the data file (2 min)

1. Evy shared **`dev.csv`** with you on Google Drive. Open it in Drive and click **Download** (⋮ menu → Download).
2. Move the downloaded `dev.csv` into the **`data`** folder inside `mse226-project` (next to the `README.md` that's
   already in there).
3. ⚠️ **Only `dev.csv`.** If you ever see `holdout.csv`, don't open it, download it, or give it to Claude. The handout
   says the holdout must stay untouched until Part 2.

Your folder should look like this:

```
mse226-project/
  data/
    dev.csv        <- you added this
    README.md
  src/  scripts/  results/  reviews/  DECISIONS.md  README.md ...
```

## Step 3: Start Claude (2 min)

1. Open the **Claude desktop app** and switch to **Cowork** (top of the window).
2. When it asks for a folder, choose your **`mse226-project`** folder. (Or click the folder icon by the message box.)
3. Copy **everything in the grey box in Step 4** and paste it as your first message. Press Enter.

## Step 4: The prompt (copy all of it)

```
I'm Person B (Alex) on a 3-person MS&E 226 (Stanford, Fundamentals of Data Science) project, Part 1. My selected folder
is our team repo, downloaded from GitHub as a ZIP (it may not be a git repo; that's fine, don't run git). My teammates already finished the setup and the models. I need you to redo my part so it matches their
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

WHEN FINISHED
- Make a folder alex_deliverables/ in my selected folder and COPY into it every file you created or changed, keeping
  their paths (alex_deliverables/scripts/06_eda.py, alex_deliverables/results/eda/..., alex_deliverables/figures/fig1.pdf).
  Do NOT copy anything from data/.
- Save the LaTeX from Task 2 as alex_deliverables/overleaf_section5.tex and the Task 4 entry as
  alex_deliverables/overleaf_section8.tex.
- Zip it as alex_deliverables.zip and give me a short list of what's inside.
```

## Step 5: While Claude works (20-40 min, mostly waiting)

- **Answer its questions.** It will ask which AI tools and versions you used, and for one suggestion you accepted or
  rejected, for Section 8.
- **Look at what it shows you.** You'll see the tables and Figure 1 before it writes text. Ask it to explain anything you
  don't follow. You'll need to explain every number and plot in the oral presentation.
- If it asks to open `holdout.csv` or the raw CMS file: **say no.**

## Step 6: Hand in your work (10 min)

**A. Send the code to Evy**

1. In your `mse226-project` folder, find **`alex_deliverables.zip`**.
2. Send it to Evy (Slack, email or Drive). Evy commits it to GitHub. You're done with GitHub.

**B. Put your text in Overleaf**

1. Open our Overleaf project (Evy will share the link). Click **main.tex** on the left.
2. Find **`\section{Data description and exploration}`** (Cmd+F, search "Data description").
3. Select from `\textbf{(a) Data collection.}` down to the end of the `(e) Missing values.` paragraph (just before
   `\section{Prediction}`). Replace it with everything in **`alex_deliverables/overleaf_section5.tex`** (open it in
   TextEdit, select all, copy).
4. In Section 6(d), replace the OLS bullet's text, which ends in `\TODO{Alex}`, with the OLS bullet from the same file.
5. In **Section 8 (AI tools)**, paste your entry from **`overleaf_section8.tex`** above Evy's.
6. In the author line near the top, replace `Alex \TODO{last name}` with your full name.
7. **Upload the figure:** on the left file panel click the **upload icon** (arrow ↑) → **New folder** first, named
   `figures`. Then select that folder → upload → choose **`alex_deliverables/figures/fig1.pdf`**.
8. Near the bottom, under `\section*{Figures}`, delete the `\TODO{...}` line and paste the
   `\includegraphics` line from your file.
9. Click **Recompile** (green button). If you see red errors, copy the error message into Claude and ask it to fix the
   LaTeX.
10. Tell Evy and Emily it's in. The report is over the page limit, so we'll trim together.

## Final checklist

- [ ] Claude ran `scripts/06_eda.py` with no errors, and you saw the tables and Figure 1.
- [ ] Nothing from `data/` is inside `alex_deliverables.zip`.
- [ ] You can explain each plot and number in your own words.
- [ ] Sections 5a–5e, the OLS bullet, your Section 8 entry, your name and Figure 1 are all in Overleaf, and it compiles.
- [ ] You sent `alex_deliverables.zip` to Evy.

Questions → message Evy.
