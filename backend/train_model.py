"""
=============================================================
Solar Power Prediction — Model Training Script
=============================================================

Trains Random Forest, XGBoost, and LightGBM on the
Enhanced Solar Dataset, evaluates all three, saves the
best model + scaler to app/trained_models/.

Features used (matching the live API):
    temperature  → AMBIENT_TEMPERATURE
    humidity     → RH2M
    pressure     → PS
    wind_speed   → WS10M
    irradiation  → IRRADIATION

Target: DC_POWER (watts)

Usage:
    cd backend
    python train_model.py
=============================================================
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────
DATASET_PATH   = "app/dataset/Enhanced_Solar_Dataset.csv"
OUTPUT_DIR     = "app/trained_models"
MODEL_PATH     = os.path.join(OUTPUT_DIR, "random_forest.pkl")
SCALER_PATH    = os.path.join(OUTPUT_DIR, "scaler.pkl")
REPORT_PATH    = os.path.join(OUTPUT_DIR, "model_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SOLAR POWER PREDICTION — MODEL TRAINING")
print("=" * 60)
print(f"\n[1/6] Loading dataset from: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION & CLEANING
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Selecting features & cleaning data ...")

# API feature → dataset column mapping
FEATURE_COLS = [
    "AMBIENT_TEMPERATURE",   # → temperature
    "RH2M",                  # → humidity
    "PS",                    # → pressure
    "WS10M",                 # → wind_speed
    "IRRADIATION",           # → irradiation
]
TARGET_COL = "DC_POWER"

# Drop rows with negative DC_POWER (data artifacts)
before = len(df)
df = df[df[TARGET_COL] >= 0]
print(f"      Removed {before - len(df):,} rows with negative DC_POWER")

# Select only needed columns
df = df[FEATURE_COLS + [TARGET_COL]].dropna()
print(f"      Final dataset: {len(df):,} rows")
print(f"      Target stats  ->  min: {df[TARGET_COL].min():.2f}  "
      f"max: {df[TARGET_COL].max():.2f}  "
      f"mean: {df[TARGET_COL].mean():.2f}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  &  SCALING
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Splitting data (80/20) and scaling features ...")

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"      Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# 4. HELPER — MAPE
# ─────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    print(f"      {name:<20} MAE={mae:>8.2f}  RMSE={rmse:>8.2f}  "
          f"R²={r2:.4f}  MAPE={mp:.2f}%")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "mape": mp}

# ─────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Training models ...\n")
results = []
trained_models = {}

# ---------- Random Forest ----------
print("  ▶ Random Forest ...")
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train_sc, y_train)
print(f"      Trained in {time.time()-t0:.1f}s")
rf_pred = np.clip(rf.predict(X_test_sc), 0, None)
res = evaluate("Random Forest", y_test, rf_pred)
results.append(res)
trained_models["Random Forest"] = rf

# ---------- XGBoost ----------
print("\n  ▶ XGBoost ...")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train_sc, y_train)
print(f"      Trained in {time.time()-t0:.1f}s")
xgb_pred = np.clip(xgb_model.predict(X_test_sc), 0, None)
res = evaluate("XGBoost", y_test, xgb_pred)
results.append(res)
trained_models["XGBoost"] = xgb_model

# ---------- LightGBM ----------
print("\n  ▶ LightGBM ...")
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)
lgb_model.fit(X_train_sc, y_train)
print(f"      Trained in {time.time()-t0:.1f}s")
lgb_pred = np.clip(lgb_model.predict(X_test_sc), 0, None)
res = evaluate("LightGBM", y_test, lgb_pred)
results.append(res)
trained_models["LightGBM"] = lgb_model

# ─────────────────────────────────────────────────────────────
# 6. SELECT BEST MODEL & SAVE
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Selecting best model by R² score ...")

best = max(results, key=lambda r: r["r2"])
best_model = trained_models[best["name"]]

print(f"  [WINNER] {best['name']}")
print(f"     MAE={best['mae']:.2f}  RMSE={best['rmse']:.2f}  "
      f"R²={best['r2']:.4f}  MAPE={best['mape']:.2f}%")

print(f"\n[6/6] Saving model + scaler ...")
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"  [OK] Model  saved -> {MODEL_PATH}")
print(f"  [OK] Scaler saved -> {SCALER_PATH}")

# ─────────────────────────────────────────────────────────────
# 7. WRITE REPORT
# ─────────────────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "  SOLAR POWER PREDICTION — MODEL COMPARISON REPORT",
    "=" * 60,
    f"  Dataset    : {DATASET_PATH}",
    f"  Rows used  : {len(df):,}",
    f"  Features   : {', '.join(FEATURE_COLS)}",
    f"  Target     : {TARGET_COL}",
    "",
    f"  {'Model':<20} {'MAE':>8} {'RMSE':>9} {'R²':>8} {'MAPE':>8}",
    "  " + "-" * 56,
]
for r in results:
    winner_mark = " <- BEST" if r["name"] == best["name"] else ""
    report_lines.append(
        f"  {r['name']:<20} {r['mae']:>8.2f} {r['rmse']:>9.2f} "
        f"{r['r2']:>8.4f} {r['mape']:>7.2f}%{winner_mark}"
    )
report_lines += [
    "",
    f"  Saved as   : {MODEL_PATH}",
    f"  Best Model : {best['name']}",
    "=" * 60,
]
report_text = "\n".join(report_lines)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"  [OK] Report  saved -> {REPORT_PATH}")
print("\n" + report_text)
print("Training complete! Start the API with:")
print("   uvicorn app.main:app --reload")
