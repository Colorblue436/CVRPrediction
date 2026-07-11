"""
server.py
=========
Flask API server for Cormeum CVD prediction.
Run: python server.py
Endpoint: POST http://localhost:5050/predict
"""

import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend dir to path so we can import sibling modules
sys.path.insert(0, __import__("os").path.dirname(__file__))

from imputer_clean import impute_input
from predictor import predict_risk

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend (localhost:8080)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Cormeum CVD API"})


# ---------------------------------------------------------------------------
# /predict  — main endpoint
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with patient inputs. Any missing optional field is
    imputed automatically. Returns risk prediction + imputed field list.

    Expected fields (all optional except age, gender, height, weight):
      age         int   years
      gender      str   "male" | "female"
      height      float cm
      weight      float kg
      ap_hi       float mmHg  (systolic)   — optional
      ap_lo       float mmHg  (diastolic)  — optional
      cholesterol int   1/2/3              — optional
      glucose     int   1/2/3              — optional
      smoke       bool                    — optional
      alco        bool                    — optional
      active      bool                    — optional
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # ── Validate required fields ──────────────────────────────────────
        required = ["age", "gender", "height", "weight"]
        missing_required = [f for f in required if data.get(f) in (None, "")]
        if missing_required:
            return jsonify({
                "error": f"Missing required fields: {missing_required}"
            }), 422

        # ── Normalise types ───────────────────────────────────────────────
        def to_float_or_none(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        normalised = {
            "age":         to_float_or_none(data["age"]),
            "gender":      str(data.get("gender", "male")).lower(),
            "height":      to_float_or_none(data["height"]),
            "weight":      to_float_or_none(data["weight"]),
            "ap_hi":       to_float_or_none(data.get("ap_hi")),
            "ap_lo":       to_float_or_none(data.get("ap_lo")),
            "cholesterol": to_float_or_none(data.get("cholesterol")),
            "glucose":     to_float_or_none(data.get("glucose")),
            "smoke":       bool(data.get("smoke", False)),
            "alco":        bool(data.get("alco", False)),
            "active":      bool(data.get("active", True)),
        }

        # ── Step 1: Impute missing values ─────────────────────────────────
        # Build the dict that the imputer understands (maps frontend names
        # to pipeline column names where they differ).
        imputer_input = {
            "age":         normalised["age"] * 365.25 if normalised["age"] else None,
            "height":      normalised["height"],
            "weight":      normalised["weight"],
            "ap_hi":       normalised["ap_hi"],
            "ap_lo":       normalised["ap_lo"],
            "cholesterol": normalised["cholesterol"],
            "gluc":        normalised["glucose"],
            "smoke":       1.0 if normalised["smoke"] else 0.0,
            "alco":        1.0 if normalised["alco"] else 0.0,
            "active":      1.0 if normalised["active"] else 0.0,
            "gender":      2.0 if normalised["gender"] == "male" else 1.0,
        }

        imputed_dict, imputed_fields = impute_input(imputer_input)

        # Map back imputed pipeline names → frontend-friendly names
        field_display_map = {
            "ap_hi": "Systolic BP", "ap_lo": "Diastolic BP",
            "cholesterol": "Cholesterol", "gluc": "Glucose",
            "smoke": "Smoking", "alco": "Alcohol", "active": "Activity",
        }
        imputed_display = [field_display_map.get(f, f) for f in imputed_fields]

        # ── Step 2: Predict ───────────────────────────────────────────────
        # Re-merge imputed values back into normalised dict for predictor
        normalised["ap_hi"]       = imputed_dict.get("ap_hi", 120)
        normalised["ap_lo"]       = imputed_dict.get("ap_lo", 80)
        normalised["cholesterol"] = imputed_dict.get("cholesterol", 1)
        normalised["glucose"]     = imputed_dict.get("gluc", 1)

        result = predict_risk(normalised)

        # ── Build response ────────────────────────────────────────────────
        response = {
            "risk_score":     result["risk_score"],
            "probability":    round(result["probability"], 4),
            "band":           result["band"],
            "badge_label":    result["badge_label"],
            "model_used":     result["model_used"],
            "bmi":            result["bmi"],
            "bmi_class":      result["bmi_class"],
            "bp_category":    result["bp_category"],
            "factors":        result["factors"],
            "imputed_fields": imputed_display,
            "used_defaults":  len(imputed_display) > 0,
            "summary": {
                "age":         normalised["age"],
                "gender":      normalised["gender"],
                "height":      normalised["height"],
                "weight":      normalised["weight"],
                "ap_hi":       normalised["ap_hi"],
                "ap_lo":       normalised["ap_lo"],
                "cholesterol": int(normalised["cholesterol"]),
                "glucose":     int(normalised["glucose"]),
                "smoke":       normalised["smoke"],
                "alco":        normalised["alco"],
                "active":      normalised["active"],
            }
        }

        # Log summary to console
        print(
            f"\n[Predict] age={normalised['age']:.0f}y "
            f"gender={normalised['gender']} "
            f"bmi={result['bmi']} "
            f"score={result['risk_score']} ({result['band']}) "
            f"model={result['model_used']} "
            f"imputed={imputed_display}",
            flush=True
        )

        return jsonify(response), 200

    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR] /predict failed:\n{tb}", flush=True)
        return jsonify({"error": "Internal server error", "detail": tb}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print(" Cormeum CVD Prediction API")
    print(" POST http://localhost:5050/predict")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)
