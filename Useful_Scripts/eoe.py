from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Predict import predict_batch_from_excel_with_score
from Predict.config import DEFAULT_DATE


# Edit these paths/values and run the script
EXCEL_PATH = Path(r"C:\Users\CharlieCampbell\Downloads\Section Worksheet all Games (10).xlsx")
OUTPUT_XLSX = Path("Data/Processed/AZ.xlsx")
EVENT_DATE = DEFAULT_DATE


if __name__ == "__main__":
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    _ = predict_batch_from_excel_with_score(
        excel_path=EXCEL_PATH,
        output_csv=OUTPUT_XLSX,
        event_date=EVENT_DATE,
    )
    print(f"[✓] Saved predictions with Score → {OUTPUT_XLSX}")
