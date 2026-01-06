#!/usr/bin/env python3
"""
Run batch predictions for a UTB Section Worksheet.
"""

import sys
from pathlib import Path
from datetime import datetime

# Make sure Bull_Model folder is in Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Predict.utb_batch_excel import predict_batch_from_excel
# If using score version:
# from Predict.utb_batch_excel import predict_batch_from_excel_with_score as predict_batch_from_excel

# -------- User Config --------

INPUT_SECTION_XLSX = Path(r"C:\Users\CharlieCampbell\Downloads\Section Worksheet (13).xlsx")

OUTPUT_FILE = (
    Path(r"Predictions")
    / f"UTB_Week_3_{datetime.now():%Y%m%d_%H%M}.xlsx"
)

EVENT_DATE = "2025-12-09"

# -------- Main --------

def main():
    print("\n[RUN] UTB Section Worksheet Prediction...\n")

    results = predict_batch_from_excel(
        excel_path=INPUT_SECTION_XLSX,
        output_csv=OUTPUT_FILE,
        event_date=EVENT_DATE,
    )

    print("\n[✓] COMPLETE")
    print(f"Rows processed: {len(results)}")
    print(f"Saved results: {OUTPUT_FILE.resolve()}\n")


if __name__ == "__main__":
    main()
