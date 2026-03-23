"""
Quick Retrain: Fix numpy pickle incompatibility
Tries to load real data; falls back to dummy model if data not found.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

DATA_PATH = "urja_nidhi_dataset_100k.csv"
MODEL_OUT = "urja_nidhi_ensemble_model.pkl"
COLUMNS_OUT = "model_feature_columns.json"
TARGET = "daily_biogas_yield"

SENSOR_FEATURES = [
    "waste_quantity", "cn_ratio", "moisture_level", "temperature", "ph",
    "retention_time", "gas_flow_rate", "methane_concentration",
    "ambient_temperature", "ambient_humidity", "nitrogen_concentration",
    "phosphorus_concentration", "potassium_concentration", "microbial_activity",
    "soil_n_requirement", "manure_equivalent_n", "external_fertilizer_required",
    "waste_collection_cost", "digester_operating_cost",
    "waste_type", "pre_treatment", "crop_type",
]

def train_real_model():
    """Train from CSV if it exists"""
    if not os.path.exists(DATA_PATH):
        return False
    
    print(f"✅ Found {DATA_PATH}, training real model...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    
    # Keep only sensor features + target
    keep = [c for c in SENSOR_FEATURES if c in df.columns] + [TARGET]
    df = df[keep]
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    feature_columns = list(X_train.columns)
    
    # Train ensemble
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    ridge = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=ridge,
        n_jobs=2, passthrough=False
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("stack", stacking)
    ])
    
    print("   Training...")
    model.fit(X_train, y_train)
    
    # Save
    joblib.dump(model, MODEL_OUT)
    with open(COLUMNS_OUT, "w") as f:
        json.dump(feature_columns, f, indent=2)
    
    print(f"✅ Model saved → {MODEL_OUT}")
    print(f"✅ Columns saved → {COLUMNS_OUT}")
    return True

def train_dummy_model():
    """Create a minimal working model"""
    print("⚠️  No training data found, creating dummy model...")
    
    feature_columns = [
        "waste_quantity", "cn_ratio", "moisture_level", "temperature", "ph",
        "retention_time", "gas_flow_rate", "methane_concentration",
        "ambient_temperature", "ambient_humidity", "nitrogen_concentration",
        "phosphorus_concentration", "potassium_concentration", "microbial_activity",
        "soil_n_requirement", "manure_equivalent_n", "external_fertilizer_required",
        "waste_collection_cost", "digester_operating_cost", "waste_type_poultry_litter",
        "waste_type_raw", "pre_treatment_fermented", "pre_treatment_raw", "crop_type_pulses", "crop_type_soybean"
    ]
    
    # Create fake training data
    n_samples = 1000
    X_fake = np.random.randn(n_samples, len(feature_columns))
    y_fake = np.random.randn(n_samples) * 50 + 100  # Fake biogas yields
    
    # Train simple model
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
    ridge = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=ridge,
        n_jobs=2, passthrough=False
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("stack", stacking)
    ])
    
    print("   Training dummy ensemble...")
    model.fit(X_fake, y_fake)
    
    # Save
    joblib.dump(model, MODEL_OUT)
    with open(COLUMNS_OUT, "w") as f:
        json.dump(feature_columns, f, indent=2)
    
    print(f"✅ Dummy model saved → {MODEL_OUT}")
    print(f"✅ Feature columns saved → {COLUMNS_OUT}")
    print("\n⚠️  PLACEHOLDER MODEL — Replace with real training when data is available!")

if __name__ == "__main__":
    try:
        if not train_real_model():
            train_dummy_model()
        print("\n✅ Retrain complete! Restart Flask app.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
