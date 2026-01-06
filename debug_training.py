#!/usr/bin/env python3
# debug_training.py

import sys
from pathlib import Path

# Add Predict to path
PREDICT_DIR = Path(__file__).resolve().parent / "Predict"
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))

from Predict.feature_engineering import solo_data_pull
import pandas as pd

print("Testing feature engineering import...")
print(f"PREDICT_DIR: {PREDICT_DIR}")
print(f"sys.path[0]: {sys.path[0]}")

# Load data
print("Loading data...")
final_data = pd.read_csv('Data/Processed/final_data.csv', parse_dates=['event_start_date'], low_memory=False)
print(f"Data shape: {final_data.shape}")

# Test feature engineering
print("Testing feature engineering...")
sample_row = final_data.loc[final_data["event_start_date"] < pd.Timestamp("2025-08-18"), ["rider_id", "bull_id"]].dropna(subset=["bull_id", "rider_id"]).head(1)

if not sample_row.empty:
    sample_rider = int(sample_row.iloc[0].rider_id)
    sample_bull = int(sample_row.iloc[0].bull_id)
    
    print(f"Sample rider_id: {sample_rider}, bull_id: {sample_bull}")
    
    features = solo_data_pull(final_data, sample_rider, sample_bull, start_date=pd.Timestamp("2025-08-18"))
    print(f"Features generated: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")
    
    # Check if expected features are present
    expected_features = ['few_rides_flag', 'few_rides_flag_hand']
    for feature in expected_features:
        if feature in features.columns:
            print(f"✓ {feature} is present")
        else:
            print(f"✗ {feature} is missing")
else:
    print("No sample rows found")

print("Done!")
