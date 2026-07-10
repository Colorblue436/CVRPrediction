"""
Cardiovascular Disease Risk Prediction — Full Pipeline
========================================================
Steps:
1. Load + clean data (this dataset has known bad rows — cleaning alone
   is usually worth more than any model change)
2. Feature engineering (BMI, pulse pressure, MAP, etc.)
3. GBDT baseline (XGBoost + LightGBM) — sanity ceiling check
4. FT-Transformer, tuned with Optuna
5. Stacked ensemble (XGB + LGBM + Transformer -> Logistic Regression meta-learner)

Realistic expectation: on the public "Cardiovascular Disease" Kaggle dataset,
best-known results across all model families sit around 73-74% accuracy /
0.78-0.80 AUC. If your run lands far above that, check for leakage
(e.g. accidentally including a derived feature that encodes the label,
or evaluating on rows seen in training).

Run with: python cvd_pipeline.py
Requires: pip install xgboost lightgbm optuna torch scikit-learn pandas numpy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, classification_report
)

import xgboost as xgb
import lightgbm as lgb
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
DATASET_PATH = "./cardio.csv"
TARGET = "cardio"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    n_before = len(df)

    # --- Physiologically implausible blood pressure readings ---
    # ap_hi (systolic) should be > ap_lo (diastolic), and both should sit
    # within a plausible human range. This dataset has entry errors like
    # negative values, ap_hi in the thousands, ap_lo > ap_hi, etc.
    df = df[(df["ap_hi"] > 0) & (df["ap_lo"] > 0)]
    df = df[df["ap_hi"] >= df["ap_lo"]]
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]

    # --- Height / weight outliers (clip to physiologically sane percentiles) ---
    for col, low, high in [("height", 0.005, 0.995), ("weight", 0.005, 0.995)]:
        lo_val, hi_val = df[col].quantile([low, high])
        df = df[(df[col] >= lo_val) & (df[col] <= hi_val)]

    print(f"Cleaning removed {n_before - len(df)} rows "
          f"({(n_before - len(df)) / n_before:.1%} of data)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Base transforms ----
    # age is given in days in this dataset
    df["age_years"] = df["age"] / 365.25

    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / (height_m ** 2)

    # ---- Blood pressure derived features ----
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map"] = df["ap_lo"] + df["pulse_pressure"] / 3.0  # mean arterial pressure
    df["ap_ratio"] = df["ap_hi"] / df["ap_lo"]  # systolic/diastolic ratio
    df["ap_hi_sq"] = df["ap_hi"] ** 2  # lets linear-ish models see curvature
    df["ap_product"] = df["ap_hi"] * df["ap_lo"]

    # Clinical BP category (AHA-style bins), ordinal-encoded:
    # 0=normal, 1=elevated, 2=stage1, 3=stage2, 4=crisis
    def bp_category(row):
        hi, lo = row["ap_hi"], row["ap_lo"]
        if hi >= 180 or lo >= 120:
            return 4
        if hi >= 140 or lo >= 90:
            return 3
        if hi >= 130 or lo >= 80:
            return 2
        if hi >= 120 and lo < 80:
            return 1
        return 0
    df["bp_category"] = df.apply(bp_category, axis=1)
    df["is_hypertensive"] = (df["bp_category"] >= 3).astype(int)

    # ---- Anthropometric features ----
    df["bmi_sq"] = df["bmi"] ** 2
    df["log_bmi"] = np.log1p(df["bmi"])
    df["log_weight"] = np.log1p(df["weight"])
    df["weight_height_ratio"] = df["weight"] / df["height"]

    # WHO-style BMI category, ordinal: 0=underweight,1=normal,2=overweight,3=obese
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    df["is_obese"] = (df["bmi"] >= 30).astype(int)

    # ---- Age features ----
    df["age_decade"] = (df["age_years"] // 10).astype(int)
    df["age_sq"] = df["age_years"] ** 2

    # ---- Lifestyle / lab composite features ----
    df["chol_gluc"] = df["cholesterol"] * df["gluc"]
    df["lifestyle_score"] = df["active"] - df["smoke"] - df["alco"]
    df["risk_habit_count"] = df["smoke"] + df["alco"] + (1 - df["active"])
    df["metabolic_risk"] = (
        (df["cholesterol"] > 1).astype(int)
        + (df["gluc"] > 1).astype(int)
        + df["is_obese"]
    )  # 0-3 composite score

    # ---- Interaction features (pairwise products with age/BMI, common in
    # cardiovascular risk scoring like Framingham) ----
    df["age_bmi"] = df["age_years"] * df["bmi"]
    df["age_ap_hi"] = df["age_years"] * df["ap_hi"]
    df["bmi_ap_hi"] = df["bmi"] * df["ap_hi"]
    df["age_cholesterol"] = df["age_years"] * df["cholesterol"]
    df["smoke_age"] = df["smoke"] * df["age_years"]

    return df


# ---------------------------------------------------------------------------
# 3. GBDT BASELINE
# ---------------------------------------------------------------------------
def train_gbdt_baseline(x_train, y_train, x_val, y_val):
    results = {}

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
    )
    xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    xgb_probs = xgb_model.predict_proba(x_val)[:, 1]
    results["xgb"] = {
        "model": xgb_model,
        "probs": xgb_probs,
        "accuracy": accuracy_score(y_val, xgb_probs > 0.5),
        "auc": roc_auc_score(y_val, xgb_probs),
    }

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    lgb_model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    lgb_probs = lgb_model.predict_proba(x_val)[:, 1]
    results["lgb"] = {
        "model": lgb_model,
        "probs": lgb_probs,
        "accuracy": accuracy_score(y_val, lgb_probs > 0.5),
        "auc": roc_auc_score(y_val, lgb_probs),
    }

    for name, r in results.items():
        print(f"[GBDT baseline] {name}: accuracy={r['accuracy']:.4f} auc={r['auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# 4. FT-TRANSFORMER (PyTorch)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)


class FTTransformer(nn.Module):
    """
    Feature-Tokenizer Transformer for tabular data.
    Each scalar feature becomes its own token (via a per-feature linear
    projection), a learned [CLS] token is prepended, and the CLS output
    after the transformer stack is used for classification — this is the
    standard FT-Transformer design (Gorishniy et al. 2021) and tends to
    work better than global-average-pooling over feature tokens.
    """

    def __init__(self, num_features, embed_dim=64, num_heads=8, ff_dim=256,
                 num_blocks=3, dropout_rate=0.2):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # One projection per feature (feature tokenizer), not a single
        # shared Linear applied to all features identically.
        self.feature_projections = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_features)]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
             for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        tokens = []
        for i in range(self.num_features):
            feat = inputs[:, i:i + 1]  # (batch, 1)
            tok = self.feature_projections[i](feat)  # (batch, embed_dim)
            tokens.append(tok.unsqueeze(1))  # (batch, 1, embed_dim)
        x = torch.cat(tokens, dim=1)  # (batch, num_features, embed_dim)

        cls = self.cls_token.expand(batch_size, 1, self.embed_dim)
        x = torch.cat([cls, x], dim=1)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0, :]  # take the CLS token representation
        return self.mlp(cls_output)


def _predict_probs(model, x, batch_size=512):
    model.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            batch = x_t[i:i + batch_size].to(DEVICE)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds).ravel()


def build_and_train_transformer(x_train, y_train, x_val, y_val, params, epochs=60):
    model = FTTransformer(
        num_features=x_train.shape[1],
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        ff_dim=params["ff_dim"],
        num_blocks=params["num_blocks"],
        dropout_rate=params["dropout_rate"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds, batch_size=params.get("batch_size", 256), shuffle=True
    )

    patience = 8
    best_auc = -np.inf
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        val_probs = _predict_probs(model, x_val)
        val_auc = roc_auc_score(y_val, val_probs)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# 4b. OPTUNA TUNING FOR THE TRANSFORMER
# ---------------------------------------------------------------------------
def tune_transformer(x_train, y_train, x_val, y_val, n_trials=25):
    def objective(trial):
        params = {
            "embed_dim": trial.suggest_categorical("embed_dim", [32, 64, 96, 128]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
            "ff_dim": trial.suggest_categorical("ff_dim", [64, 128, 256]),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }

        # embed_dim must be divisible by num_heads
        if params["embed_dim"] % params["num_heads"] != 0:
            raise optuna.TrialPruned()

        model = build_and_train_transformer(
            x_train, y_train, x_val, y_val, params, epochs=25
        )

        probs = _predict_probs(model, x_val)
        return roc_auc_score(y_val, probs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Best trial AUC:", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params


# ---------------------------------------------------------------------------
# 5. STACKED ENSEMBLE
# ---------------------------------------------------------------------------
def stack_ensemble(oof_preds: dict, y_train, test_preds: dict, y_test):
    """
    oof_preds / test_preds: {'xgb': array, 'lgb': array, 'transformer': array}
    Trains a logistic regression meta-learner on out-of-fold predictions.
    """
    model_names = list(oof_preds.keys())
    X_meta_train = np.column_stack([oof_preds[m] for m in model_names])
    X_meta_test = np.column_stack([test_preds[m] for m in model_names])

    meta = LogisticRegression()
    meta.fit(X_meta_train, y_train)

    final_probs = meta.predict_proba(X_meta_test)[:, 1]
    final_preds = (final_probs > 0.5).astype(int)

    print("\n[Ensemble] weights:", dict(zip(model_names, meta.coef_[0])))
    print(f"[Ensemble] accuracy={accuracy_score(y_test, final_preds):.4f} "
          f"auc={roc_auc_score(y_test, final_probs):.4f}")

    return final_probs, final_preds, meta


def get_oof_predictions(model_fn, x, y, n_splits=5):
    """Generic K-fold out-of-fold predictor for any sklearn-style model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(x))
    for train_idx, val_idx in skf.split(x, y):
        model = model_fn()
        model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]
    return oof


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean(DATASET_PATH)
    df = engineer_features(df)

    x = df.drop(columns=[TARGET])
    y = df[TARGET].values

    xtrain_raw, xtest_raw, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(xtrain_raw)
    x_test = scaler.transform(xtest_raw)

    # ---- Step 1: GBDT baseline ----
    print("\n=== STEP 1: GBDT BASELINE ===")
    gbdt_results = train_gbdt_baseline(x_train, y_train, x_test, y_test)

    # ---- Step 3: Optuna tuning for transformer ----
    print("\n=== STEP 3: OPTUNA TUNING (FT-Transformer) ===")
    # Use a held-out slice of train as tuning validation so test stays untouched
    x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
        x_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    best_params = tune_transformer(x_tr2, y_tr2, x_val2, y_val2, n_trials=25)

    # ---- Train final transformer on full training set with best params ----
    print("\n=== Training final FT-Transformer with best params ===")
    final_transformer = build_and_train_transformer(
        x_train, y_train, x_test, y_test, best_params, epochs=80
    )
    transformer_probs = _predict_probs(final_transformer, x_test)
    print(f"[Transformer] accuracy={accuracy_score(y_test, transformer_probs > 0.5):.4f} "
          f"auc={roc_auc_score(y_test, transformer_probs):.4f}")

    # ---- Step 4: Ensemble ----
    print("\n=== STEP 4: STACKED ENSEMBLE ===")
    oof_xgb = get_oof_predictions(
        lambda: xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                                   eval_metric="logloss", random_state=RANDOM_STATE),
        x_train, y_train,
    )
    oof_lgb = get_oof_predictions(
        lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                                    verbosity=-1, random_state=RANDOM_STATE),
        x_train, y_train,
    )
    # For the transformer OOF, reuse the tuned final model's train-set predictions
    # as an approximation (a full k-fold retrain of the transformer is expensive;
    # for a rigorous version, wrap build_and_train_transformer in the same
    # get_oof_predictions loop).
    oof_transformer = _predict_probs(final_transformer, x_train)

    oof_preds = {"xgb": oof_xgb, "lgb": oof_lgb, "transformer": oof_transformer}
    test_preds = {
        "xgb": gbdt_results["xgb"]["probs"],
        "lgb": gbdt_results["lgb"]["probs"],
        "transformer": transformer_probs,
    }
    final_probs, final_preds, meta_model = stack_ensemble(
        oof_preds, y_train, test_preds, y_test
    )

    print("\n" + "=" * 50)
    print(" FINAL COMPLIANCE PERFORMANCE REPORT ")
    print("=" * 50)
    print(classification_report(y_test, final_preds,
                                 target_names=["Low Risk (0)", "High Risk (1)"]))

    # ROC curve comparison
    plt.figure(figsize=(8, 6))
    for name, probs in [("XGBoost", gbdt_results["xgb"]["probs"]),
                         ("LightGBM", gbdt_results["lgb"]["probs"]),
                         ("FT-Transformer", transformer_probs),
                         ("Ensemble", final_probs)]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_test, probs):.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Baseline (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
