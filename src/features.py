import pandas as pd
import numpy as np


# These are the vital signs we create trend features for
# Labs are too sparse for rolling windows — we handle them differently
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

# Sparse labs — we create "was_measured" flags for these
SPARSE_LAB_COLS = [
    "Lactate", "WBC", "Creatinine", "Bilirubin_total",
    "TroponinI", "Fibrinogen", "Platelets", "pH",
    "BaseExcess", "HCO3", "Glucose", "Potassium"
]

# Columns to drop entirely — too sparse to be useful
DROP_COLS = ["EtCO2"]


def add_was_measured_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For sparse lab columns, create a binary flag:
    1 = this lab was measured this hour
    0 = not measured (was NaN)

    Why: A doctor ordering Lactate means they were worried.
    That clinical decision is itself a prediction signal.
    """
    df = df.copy()
    for col in SPARSE_LAB_COLS:
        if col in df.columns:
            df[f"{col}_measured"] = df[col].notna().astype(int)
    return df


def forward_fill_labs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient, carry lab values forward until a new
    reading arrives. Labs aren't drawn every hour — forward
    fill gives the model the most recent known value.

    CRITICAL: Must be done PER PATIENT.
    Never fill across patient boundaries.
    """
    df = df.copy()

    lab_cols = SPARSE_LAB_COLS + [
        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
        "Bilirubin_direct", "Magnesium", "Phosphate",
        "PTT", "Hct", "Hgb", "SaO2", "FiO2", "PaCO2"
    ]

    existing_labs = [c for c in lab_cols if c in df.columns]

    # Forward fill within each patient only
    df[existing_labs] = (
        df.groupby("patient_id")[existing_labs]
        .transform(lambda x: x.ffill())
    )
    return df


def add_rolling_features(df: pd.DataFrame,
                          window: int = 6) -> pd.DataFrame:
    """
    For each vital sign, compute rolling statistics
    over the last N hours (default 6).

    Features created per vital:
    - mean   : average over window (trend direction)
    - std    : variability (instability signal)
    - min    : lowest point in window
    - max    : highest point in window
    - trend  : rate of change (current - oldest) / window

    CRITICAL: Must be done PER PATIENT.
    Hour 6 of patient A must not look at hours from patient B.
    """
    df = df.copy()

    for col in VITAL_COLS:
        if col not in df.columns:
            continue

        # Group by patient, then compute rolling stats
        grp = df.groupby("patient_id")[col]

        df[f"{col}_mean_{window}hr"] = (
            grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        )

        df[f"{col}_std_{window}hr"] = (
            grp.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        )

        df[f"{col}_min_{window}hr"] = (
            grp.transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
        )

        df[f"{col}_max_{window}hr"] = (
            grp.transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        )

        # Trend = how fast is it changing per hour?
        df[f"{col}_trend_{window}hr"] = (
            grp.transform(
                lambda x: x.diff(periods=min(window, len(x)-1))
            ) / window
        )

    return df


def add_sofa_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simplified SOFA-like score.
    SOFA (Sequential Organ Failure Assessment) is the
    clinical standard for sepsis diagnosis.

    We can't compute the full SOFA (needs data we don't have)
    but we can approximate organ dysfunction signals:
    - Resp dysfunction : Resp > 22 or O2Sat < 94
    - Liver dysfunction: Bilirubin > 2 (if available)
    - Kidney stress    : Creatinine > 1.2 (if available)
    - Circulatory      : MAP < 70

    Each component = 1 point. Higher = more organ stress.
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)

    if "Resp" in df.columns:
        score += (df["Resp"] > 22).astype(int)

    if "O2Sat" in df.columns:
        score += (df["O2Sat"] < 94).astype(int)

    if "MAP" in df.columns:
        score += (df["MAP"] < 70).astype(int)

    if "Creatinine" in df.columns:
        score += (df["Creatinine"] > 1.2).fillna(0).astype(int)

    if "Bilirubin_total" in df.columns:
        score += (df["Bilirubin_total"] > 2).fillna(0).astype(int)

    df["sofa_proxy"] = score
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no information.
    EtCO2 is 100% missing — pure noise.
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def run_feature_pipeline(df: pd.DataFrame,
                          window: int = 6) -> pd.DataFrame:
    """
    Master function — runs the full feature engineering pipeline.
    Call this one function and get back a model-ready DataFrame.

    Order matters:
    1. Drop useless columns first
    2. Add was_measured flags BEFORE forward fill
       (so flags reflect actual measurements, not filled values)
    3. Forward fill labs
    4. Add rolling features on vitals
    5. Add SOFA proxy score
    """
    print("Starting feature engineering pipeline...")
    print(f"Input shape: {df.shape}")

    print("  Step 1: Dropping useless columns...")
    df = drop_useless_columns(df)

    print("  Step 2: Adding was_measured flags...")
    df = add_was_measured_flags(df)

    print("  Step 3: Forward filling lab values...")
    df = forward_fill_labs(df)

    print(f"  Step 4: Adding rolling features (window={window}hrs)...")
    df = add_rolling_features(df, window=window)

    print("  Step 5: Adding SOFA proxy score...")
    df = add_sofa_proxy(df)

    print(f"Output shape: {df.shape}")
    print(f"New features added: {df.shape[1] - 42}")
    print("Feature engineering complete.")

    return df