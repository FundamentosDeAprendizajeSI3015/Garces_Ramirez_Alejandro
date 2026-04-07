"""
Shared utilities for the dropout prediction pipeline.

Provides a single source of truth for:
- Project paths (data, outputs, figures)
- Raw data loading with column-name sanitation
- Target column name and canonical class ordering
- Figure saving helpers

Every pipeline module (01_load_and_eda.py ... 05_compare.py) imports
from this file, so changing a path or a class order here propagates
consistently across the whole project.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# utils.py lives at <project>/src/utils.py, so the project root is its parent.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data.csv"
OUTPUTS_DIR: Path = PROJECT_ROOT / "src" / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

# Make sure output directories exist the first time any module runs.
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------
# Canonical target column after column-name sanitation.
TARGET_COL: str = "Target"

# Canonical class ordering used throughout the project.
# The order matters: "Dropout" is class 0 because it is the class of
# interest (the one we want to detect early), "Graduate" is the
# positive academic outcome, and "Enrolled" is the ambiguous middle
# group of students still in the program at the end of the observation
# window. Keeping the order fixed guarantees consistent confusion
# matrices and color palettes across every figure and report.
CLASS_ORDER: list[str] = ["Dropout", "Enrolled", "Graduate"]
CLASS_TO_INT: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_INT.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the UCI 'Predict Students' Dropout and Academic Success' dataset.

    The raw CSV from UCI uses ';' as separator, contains a BOM marker
    in the first column name, and has one column whose name ends with
    a literal tab character ('Daytime/evening attendance\\t'). This
    function normalises those quirks so downstream code can reference
    clean column names.

    Parameters
    ----------
    path : Path | str | None
        Optional override for the CSV location. Defaults to DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Dataset with 4424 rows and 37 columns (36 features + Target).
    """
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download it with:\n"
            "  pip install ucimlrepo\n"
            "  python -c \"from ucimlrepo import fetch_ucirepo; import pandas as pd; "
            "d=fetch_ucirepo(id=697); "
            "pd.concat([d.data.features,d.data.targets],axis=1)"
            ".to_csv('data.csv',sep=';',index=False)\""
        )

    # utf-8-sig strips the BOM ('\ufeff') that UCI leaves in the header.
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    # Strip whitespace/tabs from column names (fixes 'Daytime/evening attendance\t').
    df.columns = df.columns.str.strip()

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into feature matrix X and target vector y.

    Returns y as strings (the original labels). Numeric encoding is
    left to each pipeline stage so that clustering modules can ignore
    the target entirely while supervised modules can encode it as they
    see fit.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------
def save_figure(fig: plt.Figure, name: str, dpi: int = 140) -> Path:
    """
    Save a matplotlib figure to the project's figures directory.

    The file is written as PNG with tight bounding box. The function
    returns the full path so callers can log it if needed.
    """
    if not name.endswith(".png"):
        name = f"{name}.png"
    out_path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Explicit grouping of the 36 features by the conceptual category
# described in the project README. Used by the EDA module to build
# per-group summaries and by downstream modules that may want to
# weight or inspect features by group.
FEATURE_GROUPS: dict[str, list[str]] = {
    "academic_1st_sem": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
    "academic_2nd_sem": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "demographic": [
        "Marital status",
        "Nacionality",
        "Gender",
        "Age at enrollment",
        "International",
        "Displaced",
    ],
    "socioeconomic": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ],
    "enrollment": [
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Previous qualification (grade)",
        "Admission grade",
    ],
    "macroeconomic": [
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ],
}