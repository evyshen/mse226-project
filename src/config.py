"""Project-wide settings. Change OUTCOME or the column lists here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEV_CSV = DATA_DIR / "dev.csv"          # the only data file Part 1 code ever reads
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SEED = 226
N_FOLDS = 5
ID = "CMS Certification Number (CCN)"

# Provisional outcome (pending Emily's sign-off in report Sections 1-2).
OUTCOME = "Rating Cycle 1 Total Number of Health Deficiencies"

# Fixed reference date for "years certified": the CMS processing date of the file.
# A constant from the data's documentation, not estimated from the data.
REFERENCE_DATE = "2026-10-01"
