#!/usr/bin/env python3
"""
Generate predictions for all rider × bull combinations and output to Excel
with one sheet per rider, sorted by probability (highest to lowest).
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Make sure Bull_Model folder is in Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Predict.cartesian_predict import predict_cartesian
from Predict.config import DEFAULT_DATE

# -------- User Config --------

# List of riders to predict for
RIDER_NAMES = [
    "Sage Kimzey",
    "Callum Miller",
    "Lucas Divino",
    "Dalton Kasel",
    "Kaique Pacheco"
    # Add more riders as needed, or load from file
]

# List of bulls to predict for
BULL_NAMES = [
    "1737 Another Round",
    "813 Crazy Party",
    "9 eyes on me",
    "016 Fire Zone",
    "024 L.A.",
    "21 Lights Out",
    "701 Magic Potion",
    "36 Rockville",
    "F15 Socks in a box",
    "128J The Player",
    "R62 UTZ BesTex Smokestack",
    "38H- War Wagon"
    # Add more bulls as needed, or load from file
]

# Event date for predictions
EVENT_DATE = DEFAULT_DATE

# Output file path
OUTPUT_FILE = (
    Path(r"Predictions")
    / f"Knockout_Predictions_{datetime.now():%Y%m%d_%H%M}.xlsx"
)

# Optional: Load from Excel file (if INPUT_FILE is set, it overrides the lists above)
# Expected format: Excel with two sheets - 'Riders' and 'Bulls', each with a 'name' column
INPUT_FILE = None  # e.g., Path(r"C:\path\to\rider_bull_list.xlsx")


# -------- Helper Functions --------

def sanitize_sheet_name(name: str, max_length: int = 31) -> str:
    """
    Excel sheet names have restrictions:
    - Max 31 characters
    - Cannot contain: \ / ? * [ ]
    """
    # Replace invalid characters
    invalid_chars = ['\\', '/', '?', '*', '[', ']']
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length]
    
    return name


def load_riders_and_bulls_from_excel(file_path: Path) -> tuple[list[str], list[str]]:
    """Load rider and bull names from an Excel file."""
    try:
        # Try to read 'Riders' sheet
        try:
            riders_df = pd.read_excel(file_path, sheet_name='Riders')
            # Look for name column (case-insensitive)
            rider_col = None
            for col in riders_df.columns:
                if 'name' in str(col).lower() or 'rider' in str(col).lower():
                    rider_col = col
                    break
            if rider_col is None:
                rider_col = riders_df.columns[0]  # Use first column if no name column
            riders = [str(name).strip() for name in riders_df[rider_col].dropna() if str(name).strip()]
        except Exception as e:
            print(f"Warning: Could not load riders sheet: {e}")
            riders = []
        
        # Try to read 'Bulls' sheet
        try:
            bulls_df = pd.read_excel(file_path, sheet_name='Bulls')
            # Look for name column (case-insensitive)
            bull_col = None
            for col in bulls_df.columns:
                if 'name' in str(col).lower() or 'bull' in str(col).lower():
                    bull_col = col
                    break
            if bull_col is None:
                bull_col = bulls_df.columns[0]  # Use first column if no name column
            bulls = [str(name).strip() for name in bulls_df[bull_col].dropna() if str(name).strip()]
        except Exception as e:
            print(f"Warning: Could not load bulls sheet: {e}")
            bulls = []
        
        return riders, bulls
    except Exception as e:
        print(f"Error loading from {file_path}: {e}")
        return [], []


# -------- Main --------

def main():
    print("\n[RUN] Rider-Bull Likelihood Predictions...\n")
    
    # Load riders and bulls
    riders = RIDER_NAMES
    bulls = BULL_NAMES
    
    if INPUT_FILE and Path(INPUT_FILE).exists():
        print(f"Loading riders and bulls from: {INPUT_FILE}")
        file_riders, file_bulls = load_riders_and_bulls_from_excel(Path(INPUT_FILE))
        if file_riders:
            riders = file_riders
            print(f"  Loaded {len(riders)} riders from file")
        if file_bulls:
            bulls = file_bulls
            print(f"  Loaded {len(bulls)} bulls from file")
    
    if not riders:
        print("ERROR: No riders specified. Please set RIDER_NAMES or provide INPUT_FILE.")
        return
    
    if not bulls:
        print("ERROR: No bulls specified. Please set BULL_NAMES or provide INPUT_FILE.")
        return
    
    print(f"Riders: {len(riders)}")
    print(f"Bulls: {len(bulls)}")
    print(f"Total combinations: {len(riders) * len(bulls)}")
    print(f"Event date: {EVENT_DATE}")
    print()
    
    # Run predictions for all combinations
    print("Running predictions (this may take a moment)...")
    try:
        predictions_df = predict_cartesian(
            rider_names=riders,
            bull_names=bulls,
            event_date=EVENT_DATE,
            mode="fast"  # Use fast mode for efficiency
        )
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Generated {len(predictions_df)} predictions\n")
    
    # Group by rider and create Excel file
    print("Organizing results by rider...")
    
    # Verify required columns exist
    required_cols = ['rider', 'bull', 'probability']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(predictions_df.columns)}")
        return
    
    # Ensure probability_pct exists (calculate if not)
    if 'probability_pct' not in predictions_df.columns:
        predictions_df['probability_pct'] = (predictions_df['probability'] * 100).round(2)
    
    # Create output directory if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Group by rider
        for rider in sorted(predictions_df['rider'].unique()):
            rider_data = predictions_df[predictions_df['rider'] == rider].copy()
            
            # Sort by probability (highest to lowest)
            rider_data = rider_data.sort_values('probability', ascending=False).reset_index(drop=True)
            
            # Select columns for output
            rider_output = pd.DataFrame({
                'Bull': rider_data['bull'],
                'Probability (%)': rider_data['probability_pct'].round(2),
                'Probability': rider_data['probability'].round(4)
            })
            
            # Sanitize sheet name for Excel
            sheet_name = sanitize_sheet_name(rider)
            
            # Write to sheet
            rider_output.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Also create a summary sheet with all predictions
        summary_df = pd.DataFrame({
            'Rider': predictions_df['rider'],
            'Bull': predictions_df['bull'],
            'Probability (%)': predictions_df['probability_pct'].round(2),
            'Probability': predictions_df['probability'].round(4)
        })
        summary_df = summary_df.sort_values(['Rider', 'Probability'], ascending=[True, False]).reset_index(drop=True)
        summary_df.to_excel(writer, sheet_name='All Predictions', index=False)
    
    print(f"\n[✓] COMPLETE")
    print(f"Results saved to: {OUTPUT_FILE.resolve()}")
    print(f"Number of rider sheets: {len(predictions_df['rider'].unique())}")
    print(f"Total predictions: {len(predictions_df)}")
    print("\nEach rider has their own sheet with bulls sorted from most to least likely.\n")


if __name__ == "__main__":
    main()

