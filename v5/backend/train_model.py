"""
Urja Nidhi — Clean Model Training Script
=========================================
Run this in Colab or locally:
  python train_model.py

Fixes vs pals.py:
  1. Drops `timestamp` (was creating 600+ dummy columns — useless for inference)
  2. Drops all confirmed leakage columns (methane_yield, energy_equivalent,
     slurry_volume, total_solids, energy_value, fertilizer_savings, etc.)
  3. Saves the exact feature column list alongside the model so the Flask
     app can always align new inputs correctly
  4. Uses Trial 2 ensemble (RF + GB + Ridge stacking) — best architecture
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "urja_nidhi_dataset_100k.csv"   # update if in Drive
MODEL_OUT   = "urja_nidhi_ensemble_model.pkl"
COLUMNS_OUT = "model_feature_columns.json"
TARGET      = "daily_biogas_yield"

# ── Columns to ALWAYS drop ────────────────────────────────────────────────────
# These are either:
#   - identifiers / timestamps (not real features)
#   - direct derivations of the target (data leakage)
DROP_ALWAYS = [
    "timestamp",           # becomes 600+ dummy cols — useless
    "methane_yield",       # directly derived from biogas yield (leakage)
    "energy_equivalent",   # biogas * conversion factor (leakage)
    "slurry_volume",       # output of digestion, not input
    "total_solids",        # output-side variable
    "energy_value",        # downstream calculation (leakage)
    "fertilizer_savings",  # downstream calculation (leakage)
]

# ── Real input features (what an IoT sensor / farmer provides) ────────────────
SENSOR_FEATURES = [
    "waste_quantity",
    "cn_ratio",
    "moisture_level",
    "temperature",
    "ph",
    "retention_time",
    "gas_flow_rate",
    "methane_concentration",
    "ambient_temperature",
    "ambient_humidity",
    "nitrogen_concentration",
    "phosphorus_concentration",
    "potassium_concentration",
    "microbial_activity",
    "soil_n_requirement",
    "manure_equivalent_n",
    "external_fertilizer_required",
    "waste_collection_cost",
    "digester_operating_cost",
    # Categorical (will be one-hot encoded):
    "waste_type",
    "pre_treatment",
    "crop_type",
]

# ── Load & Clean ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Drop leakage + timestamp cols
drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"  Dropped leakage/timestamp cols: {drop_cols}")

# Remove auto-detected high-correlation leakage (corr > 0.98 with target)
corr = df.corr(numeric_only=True)[TARGET].abs()
auto_leak = corr[corr > 0.98].index.drop(TARGET).tolist()
if auto_leak:
    print(f"  Auto-detected leakage dropped: {auto_leak}")
    df = df.drop(columns=auto_leak)

# Keep only known sensor features + target
keep = [c for c in SENSOR_FEATURES if c in df.columns] + [TARGET]
df = df[keep]
print(f"  Final columns ({len(df.columns)}): {list(df.columns)}")

# ── Split ─────────────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encode AFTER split to prevent leakage
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test,  drop_first=True)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

feature_columns = list(X_train.columns)
print(f"\nFinal feature count: {len(feature_columns)}")
print("Features:", feature_columns)

# ── Build Ensemble (Trial 2 architecture) ─────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
gb = GradientBoostingRegressor(
    n_estimators=100, max_depth=3,
    learning_rate=0.1, random_state=42
)
ridge = Ridge(alpha=1.0)

stacking = StackingRegressor(
    estimators=[("rf", rf), ("gb", gb)],
    final_estimator=ridge,
    n_jobs=2, passthrough=False
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("stack",  stacking)
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining ensemble model...")
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\n  Test MAE:  {mae:.4f}")
print(f"  Test R²:   {r2:.4f}")

# Shuffle sanity check (R² should be ≈ 0 or negative)
y_shuffled = y_test.sample(frac=1, random_state=99).values
print(f"  Shuffle R²: {r2_score(y_shuffled, y_pred):.4f}  (should be near 0)")

# Sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"  Actual: {y_test.iloc[i]:.3f}  |  Predicted: {y_pred[i]:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_OUT)
print(f"\nModel saved → {MODEL_OUT}")

with open(COLUMNS_OUT, "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"Feature columns saved → {COLUMNS_OUT}")

print("\nDone! Copy both files to your backend/ folder.")
