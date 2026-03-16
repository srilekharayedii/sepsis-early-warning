# api/main.py

# FastAPI is the web framework — handles incoming requests
from fastapi import FastAPI, HTTPException

# Pydantic validates incoming data — makes sure HR is a number
# not a string, catches bad input before it reaches the model
from pydantic import BaseModel

# Standard libraries
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Create the FastAPI app — this IS your web server
app = FastAPI(
    title="Sepsis Early Warning API",
    description="Predicts sepsis risk 6 hours before clinical diagnosis",
    version="1.0.0"
)


# ── Load model artifacts at startup ──────────────────────────
# These load ONCE when the server starts — not on every request
# Loading a model takes ~1 second. We never want that delay
# on every prediction call.

BASE_DIR = Path(__file__).parent.parent

MODEL_PATH    = BASE_DIR / "models" / "sepsis_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_cols.pkl"
THRESHOLD_PATH= BASE_DIR / "models" / "threshold.pkl"

print(f"Loading model from {MODEL_PATH}...")

try:
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    threshold    = joblib.load(THRESHOLD_PATH)
    print(f"Model loaded. {len(feature_cols)} features. "
          f"Threshold={threshold}")
except FileNotFoundError as e:
    print(f"ERROR: Model file not found: {e}")
    print("Run training notebook first to generate model files.")
    model = None


    # ── Input data structure ──────────────────────────────────────
# This defines exactly what patient data the API accepts.
# Pydantic automatically:
#   - Validates data types (HR must be float, not string)
#   - Makes fields optional with None default (labs are sparse)
#   - Returns clear error messages for bad input

class PatientData(BaseModel):
    # Vital signs — measured every hour
    HR:   float | None = None   # Heart rate (bpm)
    O2Sat:float | None = None   # Oxygen saturation (%)
    Temp: float | None = None   # Temperature (Celsius)
    SBP:  float | None = None   # Systolic blood pressure
    MAP:  float | None = None   # Mean arterial pressure
    DBP:  float | None = None   # Diastolic blood pressure
    Resp: float | None = None   # Respiratory rate

    # Trend features — computed from rolling windows
    HR_mean_6hr:   float | None = None
    HR_trend_6hr:  float | None = None
    HR_std_6hr:    float | None = None
    Resp_mean_6hr: float | None = None
    Resp_trend_6hr:float | None = None
    MAP_mean_6hr:  float | None = None
    MAP_trend_6hr: float | None = None

    # Lab indicators
    Lactate_measured:  int | None = 0
    WBC_measured:      int | None = 0
    Creatinine_measured: int | None = 0

    # SOFA proxy
    sofa_proxy: float | None = None

    # Demographics
    Age:    float | None = None
    Gender: int   | None = None
    ICULOS: float | None = None

    class Config:
        # Allow extra fields — ignore unknown columns
        extra = "allow"


# ── Prediction endpoint ───────────────────────────────────────
# This is what gets called when someone sends patient data.
# @app.post means this accepts POST requests (sending data)
# "/predict" is the URL path

@app.post("/predict")
async def predict_sepsis(patient: PatientData):
    """
    Accepts patient vitals and returns sepsis risk score.

    Send a POST request with patient measurements.
    Returns risk score, alert flag, and risk level.
    """

    # Check model loaded successfully
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first."
        )

    # Convert incoming data to DataFrame
    # The model expects a DataFrame with specific column names
    patient_dict = patient.dict()
    patient_df   = pd.DataFrame([patient_dict])

    # Add any missing columns with 0
    # (model was trained with 84 features — we need all of them)
    for col in feature_cols:
        if col not in patient_df.columns:
            patient_df[col] = 0

    # Keep only the columns the model knows about
    # in the exact order it was trained on
    patient_df = patient_df[feature_cols]

    # Fill any remaining NaN with 0
    patient_df = patient_df.fillna(0)

    # Get probability score from model
    # predict_proba returns [prob_no_sepsis, prob_sepsis]
    # we want index [1] — probability of sepsis
    risk_score = float(
        model.predict_proba(patient_df)[0][1]
    )

    # Apply clinical threshold
    alert = risk_score >= threshold

    # Determine risk level for human readability
    if risk_score >= 0.7:
        risk_level = "CRITICAL"
    elif risk_score >= 0.4:
        risk_level = "HIGH"
    elif risk_score >= 0.1:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Return the prediction
    return {
        "risk_score":  round(risk_score, 4),
        "alert":       bool(alert),
        "risk_level":  risk_level,
        "threshold":   threshold,
        "message":     (
            "SEPSIS RISK DETECTED — Immediate assessment recommended"
            if alert else
            "Low risk — Continue standard monitoring"
        )
    }


# ── Health check endpoint ─────────────────────────────────────
# Standard in every production API.
# Lets monitoring systems check "is the server alive?"

@app.get("/health")
async def health_check():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "features":     len(feature_cols) if feature_cols else 0,
        "threshold":    threshold
    }


# ── Root endpoint ─────────────────────────────────────────────
# What you see when you visit the base URL

@app.get("/")
async def root():
    return {
        "name":        "Sepsis Early Warning API",
        "version":     "1.0.0",
        "description": "Predicts sepsis risk 6hrs before diagnosis",
        "endpoints": {
            "predict":  "POST /predict",
            "health":   "GET /health",
            "docs":     "GET /docs"
        }
    }
