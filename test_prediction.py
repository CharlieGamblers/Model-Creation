#!/usr/bin/env python3
# test_prediction.py

from Predict.predictions import predict_one

# Test prediction
print("Testing prediction with Sage Kimzey and 19H Man Hater...")
try:
    result = predict_one('Sage Kimzey', '19H Man Hater', event_date='2025-01-15')
    print('✓ Prediction successful!')
    print(f'Rider: {result["rider"]}')
    print(f'Bull: {result["bull"]}')
    print(f'Probability: {result["probability"]:.4f}')
    print(f'Base Probability: {result["base_probability"]:.4f}')
except Exception as e:
    print(f'✗ Prediction failed: {e}')
    import traceback
    traceback.print_exc()

print("Done!")
