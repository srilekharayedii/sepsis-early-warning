import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_patient_files(data_dir: str, max_patients: int = None) -> pd.DataFrame:
    """
    Load all patient .psv files from a directory into one DataFrame.

    Each row = one hour for one patient.
    Each file = one patient's complete ICU stay.

    Parameters
    ----------
    data_dir     : path to folder containing .psv files
    max_patients : load only N patients (useful for fast testing)

    Returns
    -------
    pd.DataFrame with all patients stacked vertically
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.psv"))

    if not files:
        raise FileNotFoundError(f"No .psv files found in {data_dir}")

    if max_patients:
        files = files[:max_patients]

    print(f"Loading {len(files)} patient files...")

    dfs = []
    for i, filepath in enumerate(files):
        df = pd.read_csv(filepath, sep="|")

        # Add patient ID from filename — p000001.psv becomes p000001
        df["patient_id"] = filepath.stem

        dfs.append(df)

        # Show progress every 2000 patients
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} patients loaded...")

    print(f"Combining into single DataFrame...")
    combined = pd.concat(dfs, ignore_index=True)

    print(f"\n=== Dataset Loaded ===")
    print(f"Total rows (patient-hours) : {len(combined):,}")
    print(f"Unique patients            : {combined['patient_id'].nunique():,}")
    print(f"Columns                    : {combined.shape[1]}")
    print(f"Sepsis cases (hours)       : {combined['SepsisLabel'].sum():,}")
    print(f"Sepsis rate                : {combined['SepsisLabel'].mean()*100:.2f}%")

    return combined


def get_column_groups() -> dict:
    """
    Returns column names grouped by clinical category.
    Use this everywhere instead of hardcoding column names.
    """
    return {
        "vitals": [
            "HR", "O2Sat", "Temp",
            "SBP", "MAP", "DBP", "Resp"
        ],
        "labs": [
            "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
            "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
            "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate",
            "Potassium", "Bilirubin_total", "TroponinI",
            "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"
        ],
        "demographics": [
            "Age", "Gender", "Unit1", "Unit2",
            "HospAdmTime", "ICULOS"
        ],
        "target": ["SepsisLabel"]
    }


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing missing data rate for each column.
    Sorted from most missing to least.
    Useful for deciding imputation strategy.
    """
    missing = df.isnull().mean().sort_values(ascending=False)
    result = pd.DataFrame({
        "column": missing.index,
        "missing_rate": missing.values,
        "missing_pct": (missing.values * 100).round(1)
    })
    return result[result["missing_rate"] > 0].reset_index(drop=True)