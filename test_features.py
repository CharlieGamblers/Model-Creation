#!/usr/bin/env python3
# test_features.py

from Predict.feature_engineering import solo_data_pull
import pandas as pd

# Load data
print("Loading data...")
df = pd.read_csv('Data/Processed/final_data.csv', parse_dates=['event_start_date'], low_memory=False)
print(f"Data loaded, shape: {df.shape}")

# Test feature engineering
print("Testing feature engineering...")
features = solo_data_pull(df, 8924, 1234, start_date=pd.to_datetime('2025-01-15'))
print(f"Features generated, shape: {features.shape}")
print("Feature columns:", features.columns.tolist())

# Check if expected features are present
expected_features = ['few_rides_flag', 'few_rides_flag_hand']
for feature in expected_features:
    if feature in features.columns:
        print(f"✓ {feature} is present")
    else:
        print(f"✗ {feature} is missing")

print("Done!")
