"""
predictor.py
=============
Inference layer built on top of cvd_pipeline_pytorch.py logic.

Extracts:
  - engineer_features_single()  — feature engineering for one patient row
  - CvdPredictor class          — loads/trains a model and provides predict()

Model priority:
  1. Load a pre-saved cvd_model.joblib (fastest)
  2. Train a fresh XGBoost on cardio.csv if found (one-time cost)
  3. Fall back to a calibrated logistic formula (always available)
"""

import os
import math
import numpy as np
import pandas as pd

try:
    import joblib
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_HERE, "cvd_model.joblib")
SCALER_PATH = os.path.join(_HERE, "cvd_scaler.joblib")
# Look for the cardio dataset in several common locations
DATASET_CANDIDATES = [
    os.path.join(_HERE, "cardio.csv"),
    os.path.join(_HERE, "..", "cardio.csv"),
    r"D:\Heart-Disease-Detection-main\cardio.csv",
    r"D:\Heart-Disease-Detection-main\heart_disease_uci.csv",
]


# ---------------------------------------------------------------------------
# Feature Engineering  (ported from cvd_pipeline_pytorch.py)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi", "pulse_pressure", "map", "ap_ratio", "ap_hi_sq", "ap_product",
    "bp_category", "is_hypertensive",
    "bmi_sq", "log_bmi", "log_weight", "weight_height_ratio",
    "bmi_category", "is_obese",
    "age_decade", "age_sq",
    "chol_gluc", "lifestyle_score", "risk_habit_count", "metabolic_risk",
    "age_bmi", "age_ap_hi", "bmi_ap_hi", "age_cholesterol", "smoke_age",
]


def engineer_features_single(row: dict) -> dict:
    """
    Apply the same feature engineering as cvd_pipeline_pytorch.engineer_features()
    to a single patient dictionary.
    """
    age     = float(row.get("age", 19720))       # in days
    height  = float(row.get("height", 165))
    weight  = float(row.get("weight", 72))
    ap_hi   = float(row.get("ap_hi", 120))
    ap_lo   = float(row.get("ap_lo", 80))
    chol    = float(row.get("cholesterol", 1))
    gluc    = float(row.get("gluc", 1))
    smoke   = float(row.get("smoke", 0))
    alco    = float(row.get("alco", 0))
    active  = float(row.get("active", 1))
    gender  = float(row.get("gender", 2))  # 1=female, 2=male

    age_years = age / 365.25
    height_m  = height / 100.0
    bmi       = weight / (height_m ** 2)

    pulse_pressure = ap_hi - ap_lo
    map_val        = ap_lo + pulse_pressure / 3.0
    ap_ratio       = ap_hi / max(ap_lo, 1)
    ap_hi_sq       = ap_hi ** 2
    ap_product     = ap_hi * ap_lo

    # BP category (AHA-style)
    if ap_hi >= 180 or ap_lo >= 120:
        bp_cat = 4
    elif ap_hi >= 140 or ap_lo >= 90:
        bp_cat = 3
    elif ap_hi >= 130 or ap_lo >= 80:
        bp_cat = 2
    elif ap_hi >= 120 and ap_lo < 80:
        bp_cat = 1
    else:
        bp_cat = 0

    is_hypertensive = 1 if bp_cat >= 3 else 0

    bmi_sq               = bmi ** 2
    log_bmi              = math.log1p(bmi)
    log_weight           = math.log1p(weight)
    weight_height_ratio  = weight / max(height, 1)

    if bmi < 18.5:    bmi_cat = 0
    elif bmi < 25:    bmi_cat = 1
    elif bmi < 30:    bmi_cat = 2
    else:             bmi_cat = 3

    is_obese = 1 if bmi >= 30 else 0

    age_decade = int(age_years // 10)
    age_sq     = age_years ** 2

    chol_gluc       = chol * gluc
    lifestyle_score = active - smoke - alco
    risk_habit_count = smoke + alco + (1 - active)
    metabolic_risk  = int(chol > 1) + int(gluc > 1) + is_obese

    age_bmi        = age_years * bmi
    age_ap_hi      = age_years * ap_hi
    bmi_ap_hi      = bmi * ap_hi
    age_cholesterol = age_years * chol
    smoke_age      = smoke * age_years

    return {
        "age_years": age_years, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
        "smoke": smoke, "alco": alco, "active": active,
        "bmi": bmi, "pulse_pressure": pulse_pressure, "map": map_val,
        "ap_ratio": ap_ratio, "ap_hi_sq": ap_hi_sq, "ap_product": ap_product,
        "bp_category": bp_cat, "is_hypertensive": is_hypertensive,
        "bmi_sq": bmi_sq, "log_bmi": log_bmi, "log_weight": log_weight,
        "weight_height_ratio": weight_height_ratio,
        "bmi_category": bmi_cat, "is_obese": is_obese,
        "age_decade": age_decade, "age_sq": age_sq,
        "chol_gluc": chol_gluc, "lifestyle_score": lifestyle_score,
        "risk_habit_count": risk_habit_count, "metabolic_risk": metabolic_risk,
        "age_bmi": age_bmi, "age_ap_hi": age_ap_hi, "bmi_ap_hi": bmi_ap_hi,
        "age_cholesterol": age_cholesterol, "smoke_age": smoke_age,
    }


# ---------------------------------------------------------------------------
# Logistic fallback formula  (calibrated from published CVD risk research)
# ---------------------------------------------------------------------------
def _logistic_fallback(row: dict) -> float:
    """Returns a probability in [0, 1] without any trained model."""
    age_years = float(row.get("age", 19720)) / 365.25
    bmi       = float(row.get("bmi", 25))
    ap_hi     = float(row.get("ap_hi", 120))
    ap_lo     = float(row.get("ap_lo", 80))
    chol      = float(row.get("cholesterol", 1))
    gluc      = float(row.get("gluc", 1))
    smoke     = float(row.get("smoke", 0))
    alco      = float(row.get("alco", 0))
    active    = float(row.get("active", 1))
    gender    = float(row.get("gender", 2))  # 2=male

    z = -3.8
    z += age_years * 0.042
    if gender == 2: z += 0.38            # male
    if bmi >= 25 and bmi < 30: z += 0.25
    if bmi >= 30: z += 0.60
    bp_cat = row.get("bp_category", 0)
    if bp_cat == 1: z += 0.30
    if bp_cat == 2: z += 0.75
    if bp_cat == 3: z += 1.45
    if bp_cat == 4: z += 2.30
    pp = ap_hi - ap_lo
    if pp > 50: z += (pp - 50) * 0.015
    if chol == 2: z += 0.55
    if chol == 3: z += 1.20
    if gluc == 2: z += 0.35
    if gluc == 3: z += 0.85
    if smoke:  z += 0.80
    if alco:   z += 0.25
    if active: z -= 0.30

    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Load / train the model
# ---------------------------------------------------------------------------
def _find_dataset():
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _load_and_clean(path: str) -> pd.DataFrame:
    """Minimal clean (matches cvd_pipeline_pytorch.load_and_clean)."""
    sep = ";" if path.endswith(".csv") else ","
    # Try semicolon first, then comma
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")

    # Kaggle cardio dataset uses column 'gluc'; UCI uses 'ca', 'thal', etc.
    if "cardio" not in df.columns:
        # Not the cardio dataset — cannot train on it
        return None

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[df["ap_hi"] >= df["ap_lo"]]
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]
    return df.reset_index(drop=True)


def _train_model(df: pd.DataFrame):
    """Engineer features + train XGBoost; persist model + scaler."""
    from sklearn.model_selection import train_test_split

    # Engineer features on full dataset
    records = []
    for _, row in df.iterrows():
        feat = engineer_features_single(dict(row))
        records.append(feat)
    feat_df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    y = df["cardio"].values

    X = feat_df[FEATURE_COLUMNS].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        early_stopping_rounds=25,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    from sklearn.metrics import roc_auc_score
    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"  [Predictor] XGBoost trained — val AUC: {auc:.4f}", flush=True)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler


# ---------------------------------------------------------------------------
# Public predict function
# ---------------------------------------------------------------------------
_model  = None
_scaler = None
_mode   = "fallback"  # "xgboost" | "fallback"


def _ensure_model():
    global _model, _scaler, _mode

    if _model is not None:
        return  # already loaded

    if not SKLEARN_AVAILABLE:
        _mode = "fallback"
        return

    # 1. Try loading a saved model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            _model  = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            _mode   = "xgboost"
            print("  [Predictor] Loaded saved XGBoost model.", flush=True)
            return
        except Exception as e:
            print(f"  [Predictor] Load failed ({e}), re-training.", flush=True)

    # 2. Try training on the dataset
    dataset_path = _find_dataset()
    if dataset_path:
        print(f"  [Predictor] Training on {dataset_path} …", flush=True)
        try:
            df = _load_and_clean(dataset_path)
            if df is not None and len(df) > 1000:
                _model, _scaler = _train_model(df)
                _mode = "xgboost"
                return
        except Exception as e:
            print(f"  [Predictor] Training failed ({e}), using fallback.", flush=True)

    _mode = "fallback"
    print("  [Predictor] Using calibrated logistic fallback.", flush=True)


def predict_risk(input_dict: dict) -> dict:
    """
    Full CVD risk prediction for a single patient.

    Parameters
    ----------
    input_dict : cleaned, fully-imputed dict with keys matching POPULATION_DEFAULTS

    Returns
    -------
    dict with keys:
      probability   float  0-1
      risk_score    int    0-100
      band          str    "low" | "mid" | "high"
      badge_label   str    human-readable band label
      model_used    str    "xgboost" | "fallback"
      factors       list   [{name, pct, is_risk, note}]
    """
    _ensure_model()

    # Map frontend field names → pipeline field names
    # Frontend uses age in years; pipeline expects days
    age_years = float(input_dict.get("age", 54))
    row = {
        "age":         age_years * 365.25,
        "gender":      2 if input_dict.get("gender", "male") == "male" else 1,
        "height":      float(input_dict.get("height", 165)),
        "weight":      float(input_dict.get("weight", 72)),
        "ap_hi":       float(input_dict.get("ap_hi", 120)),
        "ap_lo":       float(input_dict.get("ap_lo", 80)),
        "cholesterol": float(input_dict.get("cholesterol", 1)),
        "gluc":        float(input_dict.get("glucose", 1)),
        "smoke":       1.0 if input_dict.get("smoke") in (True, 1, "true", "1") else 0.0,
        "alco":        1.0 if input_dict.get("alco")  in (True, 1, "true", "1") else 0.0,
        "active":      1.0 if input_dict.get("active") in (True, 1, "true", "1") else 0.0,
    }

    # Engineer features
    feat = engineer_features_single(row)

    # Get probability
    if _mode == "xgboost":
        feat_arr = np.array([[feat[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
        feat_scaled = _scaler.transform(feat_arr)
        probability = float(_model.predict_proba(feat_scaled)[0][1])
    else:
        probability = _logistic_fallback({**row, **feat})

    risk_score = max(2, min(99, round(probability * 100)))

    if risk_score < 10:
        band, badge_label = "low",  "Low risk"
    elif risk_score < 25:
        band, badge_label = "mid",  "Moderate risk"
    else:
        band, badge_label = "high", "Elevated risk"

    # Contributing factors
    bp_pct_map = {0: 8, 1: 28, 2: 52, 3: 82, 4: 96}
    bp_cat = feat["bp_category"]
    factors = [
        {
            "name":    "Blood pressure",
            "pct":     bp_pct_map.get(bp_cat, 8),
            "is_risk": bp_pct_map.get(bp_cat, 8) > 20,
            "note":    f"{int(row['ap_hi'])}/{int(row['ap_lo'])} mmHg",
        },
        {
            "name":    "Age",
            "pct":     int(min(max((age_years - 18) / 62 * 100, 5), 95)),
            "is_risk": age_years >= 45,
            "note":    f"{age_years:.0f} yrs",
        },
        {
            "name":    "BMI",
            "pct":     70 if feat["bmi"] >= 30 else 38 if feat["bmi"] >= 25 else 10,
            "is_risk": feat["bmi"] >= 25,
            "note":    f"{feat['bmi']:.1f}",
        },
        {
            "name":    "Cholesterol",
            "pct":     80 if row["cholesterol"] == 3 else 44 if row["cholesterol"] == 2 else 9,
            "is_risk": row["cholesterol"] > 1,
            "note":    ["", "Normal", "Above", "Well above"][int(row["cholesterol"])],
        },
        {
            "name":    "Glucose",
            "pct":     64 if row["gluc"] == 3 else 33 if row["gluc"] == 2 else 8,
            "is_risk": row["gluc"] > 1,
            "note":    ["", "Normal", "Above", "Well above"][int(row["gluc"])],
        },
        {
            "name":    "Smoking",
            "pct":     74 if row["smoke"] else 5,
            "is_risk": bool(row["smoke"]),
            "note":    "Active smoker" if row["smoke"] else "Non-smoker",
        },
        {
            "name":    "Sex",
            "pct":     38 if row["gender"] == 2 else 18,
            "is_risk": row["gender"] == 2,
            "note":    "Male" if row["gender"] == 2 else "Female",
        },
    ]

    return {
        "probability":  probability,
        "risk_score":   risk_score,
        "band":         band,
        "badge_label":  badge_label,
        "model_used":   _mode,
        "bmi":          round(feat["bmi"], 1),
        "bmi_class":    (
            "Underweight" if feat["bmi"] < 18.5
            else "Normal" if feat["bmi"] < 25
            else "Overweight" if feat["bmi"] < 30
            else "Obese"
        ),
        "bp_category":  int(bp_cat),
        "factors":      factors,
    }
