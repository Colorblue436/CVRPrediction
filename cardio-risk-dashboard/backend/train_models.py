"""
train_models.py
===============
One-time offline training script.
Trains:
  1. XGBoost imputer models for optional fields (ap_hi, ap_lo, cholesterol, glucose)
  2. XGBoost CVD risk classifier

Run: python train_models.py
Requires: pip install xgboost scikit-learn pandas numpy joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Synthetic training data generator
# Uses the Framingham/AHA risk coefficients to
# generate a realistic 20 000-row dataset when
# no real CSV is available.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_synthetic_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    age         = rng.integers(20, 80, n).astype(float)
    gender      = rng.integers(1, 3, n).astype(float)      # 1=female, 2=male
    height      = rng.normal(168, 9, n).clip(145, 200)
    weight      = rng.normal(75, 14, n).clip(40, 160)
    bmi         = weight / (height / 100) ** 2
    ap_hi       = (110 + age * 0.5 + rng.normal(0, 12, n)).clip(80, 220)
    ap_lo       = (ap_hi * 0.63 + rng.normal(0, 8, n)).clip(50, 130)
    cholesterol = rng.choice([1, 2, 3], n, p=[0.60, 0.25, 0.15]).astype(float)
    gluc        = rng.choice([1, 2, 3], n, p=[0.70, 0.20, 0.10]).astype(float)
    smoke       = rng.choice([0, 1],   n, p=[0.80, 0.20]).astype(float)
    alco        = rng.choice([0, 1],   n, p=[0.75, 0.25]).astype(float)
    active      = rng.choice([0, 1],   n, p=[0.30, 0.70]).astype(float)

    # Logistic risk model â†’ binary label
    z = (
        -4.0
        + age         * 0.045
        + (gender==2) * 0.40
        + (bmi>=25)   * 0.25
        + (bmi>=30)   * 0.40
        + ((ap_hi>=130).astype(float)) * 0.70
        + ((ap_hi>=140).astype(float)) * 0.75
        + (ap_hi - ap_lo - 40) * 0.012
        + (cholesterol==2) * 0.55
        + (cholesterol==3) * 1.20
        + (gluc==2)        * 0.35
        + (gluc==3)        * 0.80
        + smoke            * 0.80
        + alco             * 0.25
        - active           * 0.30
    )
    prob  = 1 / (1 + np.exp(-z))
    cardio = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi.round(1), "ap_lo": ap_lo.round(1),
        "cholesterol": cholesterol, "gluc": gluc,
        "smoke": smoke, "alco": alco, "active": active,
        "cardio": cardio,
    })


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Feature engineering (mirrors predictor.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_years"]           = df["age"]
    df["bmi"]                 = df["weight"] / (df["height"] / 100) ** 2
    df["pulse_pressure"]      = df["ap_hi"] - df["ap_lo"]
    df["map"]                 = df["ap_lo"] + df["pulse_pressure"] / 3.0
    df["ap_ratio"]            = df["ap_hi"] / df["ap_lo"].clip(lower=1)
    df["ap_hi_sq"]            = df["ap_hi"] ** 2
    df["ap_product"]          = df["ap_hi"] * df["ap_lo"]
    df["bmi_sq"]              = df["bmi"] ** 2
    df["log_bmi"]             = np.log1p(df["bmi"])
    df["log_weight"]          = np.log1p(df["weight"])
    df["weight_height_ratio"] = df["weight"] / df["height"].clip(lower=1)
    df["age_sq"]              = df["age_years"] ** 2
    df["age_decade"]          = (df["age_years"] // 10).astype(int)
    df["chol_gluc"]           = df["cholesterol"] * df["gluc"]
    df["lifestyle_score"]     = df["active"] - df["smoke"] - df["alco"]
    df["risk_habit_count"]    = df["smoke"] + df["alco"] + (1 - df["active"])
    df["is_obese"]            = (df["bmi"] >= 30).astype(int)
    df["is_hypertensive"]     = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    df["metabolic_risk"]      = (
        (df["cholesterol"] > 1).astype(int)
        + (df["gluc"] > 1).astype(int)
        + df["is_obese"]
    )
    df["age_bmi"]             = df["age_years"] * df["bmi"]
    df["age_ap_hi"]           = df["age_years"] * df["ap_hi"]
    df["bmi_ap_hi"]           = df["bmi"] * df["ap_hi"]
    df["age_cholesterol"]     = df["age_years"] * df["cholesterol"]
    df["smoke_age"]           = df["smoke"] * df["age_years"]

    def bp_cat(row):
        h, l = row["ap_hi"], row["ap_lo"]
        if h >= 180 or l >= 120: return 4
        if h >= 140 or l >= 90:  return 3
        if h >= 130 or l >= 80:  return 2
        if h >= 120 and l < 80:  return 1
        return 0
    df["bp_category"] = df.apply(bp_cat, axis=1)

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


FEATURE_COLS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active",
    "bmi", "pulse_pressure", "map", "ap_ratio", "ap_hi_sq", "ap_product",
    "bp_category", "is_hypertensive",
    "bmi_sq", "log_bmi", "log_weight", "weight_height_ratio",
    "bmi_category", "is_obese",
    "age_decade", "age_sq",
    "chol_gluc", "lifestyle_score", "risk_habit_count", "metabolic_risk",
    "age_bmi", "age_ap_hi", "bmi_ap_hi", "age_cholesterol", "smoke_age",
]

# Optional fields the user might leave blank â†’ we need imputers for these
IMPUTE_TARGETS = ["ap_hi", "ap_lo", "cholesterol", "gluc"]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 â€” Train imputers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_imputers(df: pd.DataFrame):
    print("\n[1/2] Training imputer models â€¦")
    # Predictors are all raw columns except the imputation targets
    predictor_cols = [
        c for c in ["age", "gender", "height", "weight", "smoke", "alco", "active"]
        if c in df.columns
    ]

    imputers = {}
    for target in IMPUTE_TARGETS:
        if target not in df.columns:
            print(f"  Skipping {target} (not in dataset)")
            continue
        sub = df[predictor_cols + [target]].dropna()
        X = sub[predictor_cols].values
        y = sub[target].values

        reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        reg.fit(X, y)
        imputers[target] = {"model": reg, "predictor_cols": predictor_cols}
        print(f"  [OK] Imputer for '{target}' trained")

    out = os.path.join(HERE, "imputer_models.joblib")
    joblib.dump(imputers, out)
    print(f"  Saved â†’ {out}")
    return imputers


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 â€” Train classifier
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_classifier(df: pd.DataFrame):
    print("\n[2/2] Training CVD risk classifier â€¦")
    feat_df = _engineer(df)
    avail   = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[avail].values
    y = df["cardio"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
        early_stopping_rounds=30,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, probs)
    acc   = accuracy_score(y_val, probs > 0.5)
    print(f"  Val AUC: {auc:.4f}  |  Accuracy: {acc:.4f}")

    # Save
    joblib.dump(model,  os.path.join(HERE, "cvd_model.joblib"))
    joblib.dump(scaler, os.path.join(HERE, "cvd_scaler.joblib"))
    joblib.dump(avail,  os.path.join(HERE, "cvd_feature_cols.joblib"))
    print(f"  Saved â†’ cvd_model.joblib / cvd_scaler.joblib / cvd_feature_cols.joblib")
    return model, scaler, avail


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    print("=" * 55)
    print("  Cormeum â€” Model Training Pipeline")
    print("=" * 55)

    # Try to load real data, fall back to synthetic
    real_paths = [
        os.path.join(HERE, "..", "cardio.csv"),
        r"D:\Heart-Disease-Detection-main\cardio.csv",
    ]
    df = None
    for p in real_paths:
        if os.path.exists(p):
            try:
                raw = pd.read_csv(p, sep=";")
                if "cardio" in raw.columns:
                    # Kaggle cardio dataset: age is in days, rename glucâ†’gluc
                    raw["age"] = raw["age"] / 365.25
                    df = raw
                    print(f"Loaded real dataset: {p}  ({len(df)} rows)")
                    break
            except Exception as e:
                print(f"Could not load {p}: {e}")

    if df is None:
        print("No real cardio.csv found â€” generating synthetic training data â€¦")
        df = generate_synthetic_data(n=20000)
        print(f"Synthetic dataset: {len(df)} rows")

    # Ensure required columns exist
    required = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Dataset missing columns: {missing}")
        print("Falling back to synthetic data.")
        df = generate_synthetic_data(n=20000)

    train_imputers(df)
    train_classifier(df)

    print("\n[DONE] All models trained and saved.")
    print("   Restart server.py to pick up the new models.")
