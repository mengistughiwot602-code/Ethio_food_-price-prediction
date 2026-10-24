import os
import joblib
import pandas as pd
import numpy as np

print("--- Offline Diagnostics ---")
print("1. Verifying Files in Directory:")
for f in ['app.py', 'main.py', 'random_forest_model.joblib', 'wfp_food_prices_et.xlsx', 'requirements.txt']:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"  - {f}: {'Exists (' + str(size) + ' bytes)' if exists else 'MISSING'}")

print("\n2. Verifying Model Assets Loading:")
try:
    model = joblib.load("random_forest_model.joblib")
    print("  - Model loaded successfully!")
    print(f"  - Expected input features size: {len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown'}")
except Exception as e:
    print(f"  - Model failed to load: {e}")

print("\n3. Verifying Excel Dataset:")
try:
    df = pd.read_excel("wfp_food_prices_et.xlsx")
    print(f"  - Dataset loaded successfully! Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
except Exception as e:
    print(f"  - Dataset failed to load: {e}")
