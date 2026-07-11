"""
imputer_clean.py
================
Predictive imputation for optional CVD input fields.

Loads XGBoost regressors trained by train_models.py.
Falls back to evidence-based population medians if models are absent.
"""

import os
import numpy as np

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# Population-level medians / modes (Kaggle Cardiovascular Disease dataset)
POPULATION_DEFAULTS = {
    "ap_hi":       120.0,   # mmHg systolic
    "ap_lo":       80.0,    # mmHg diastolic
    "cholesterol": 1.0,     # 1=Normal
    "gluc":        1.0,     # 1=Normal
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "imputer_models.joblib")

# Module-level cache
_imputers = None


def _load_imputers():
    global _imputers
    if _imputers is not None:
        return _imputers
    if not JOBLIB_OK:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        _imputers = joblib.load(MODEL_PATH)
        print(f"  [Imputer] Loaded XGBoost imputers for: {list(_imputers.keys())}", flush=True)
        return _imputers
    except Exception as e:
        print(f"  [Imputer] Failed to load models: {e}", flush=True)
        return None


def impute_input(user_dict: dict) -> tuple:
    """
    Fill any None/NaN optional fields in user_dict using trained XGBoost regressors.

    Parameters
    ----------
    user_dict : dict with keys matching pipeline column names:
        age, gender, height, weight, ap_hi, ap_lo,
        cholesterol, gluc, smoke, alco, active

    Returns
    -------
    (complete_dict, imputed_fields)
        complete_dict  : dict with all fields filled
        imputed_fields : list of field names that were imputed
    """
    complete = {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else v)
                for k, v in user_dict.items()}
    imputed_fields = []

    missing = [col for col in POPULATION_DEFAULTS if complete.get(col) is None]
    if not missing:
        return complete, []

    imputers = _load_imputers()

    for col in missing:
        predicted = False

        # ── XGBoost model ───────────────────────────────────────────────
        if imputers and col in imputers:
            try:
                entry        = imputers[col]
                model        = entry["model"]
                pred_cols    = entry["predictor_cols"]
                feat_values  = [complete.get(c, 0) or 0 for c in pred_cols]
                import numpy as _np
                feat_arr     = _np.array([feat_values], dtype=np.float32)
                predicted_val = float(model.predict(feat_arr)[0])

                # Clamp to physiologically sane ranges
                clamp = {
                    "ap_hi":       (80,  220),
                    "ap_lo":       (50,  130),
                    "cholesterol": (1,   3),
                    "gluc":        (1,   3),
                }
                if col in clamp:
                    lo, hi = clamp[col]
                    if col in ("cholesterol", "gluc"):
                        predicted_val = round(max(lo, min(hi, predicted_val)))
                    else:
                        predicted_val = round(max(lo, min(hi, predicted_val)), 1)

                complete[col] = predicted_val
                imputed_fields.append(col)
                print(f"  [Imputer] '{col}' predicted = {predicted_val}", flush=True)
                predicted = True
            except Exception as e:
                print(f"  [Imputer] XGBoost predict failed for '{col}': {e}", flush=True)

        # ── Fallback: population median ─────────────────────────────────
        if not predicted:
            complete[col] = POPULATION_DEFAULTS[col]
            imputed_fields.append(col)
            print(f"  [Imputer] '{col}' → population default {POPULATION_DEFAULTS[col]}", flush=True)

    return complete, imputed_fields
