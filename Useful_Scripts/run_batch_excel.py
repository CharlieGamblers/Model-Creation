from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Predict import predict_batch_from_excel
from Predict.config import DEFAULT_DATE


# Edit these paths/values and run the script
EXCEL_PATH = Path(r"C:\Users\CharlieCampbell\Downloads\Section Worksheet day Games (13).xlsx")
OUTPUT_XLSX = Path("Data/Processed/predictions.xlsx")
EVENT_DATE = DEFAULT_DATE


if __name__ == "__main__":
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    results = predict_batch_from_excel(
        excel_path=EXCEL_PATH,
        output_csv=OUTPUT_XLSX,
        event_date=EVENT_DATE,
    )
    print(f"\n[✓] {len(results)} predictions saved → {OUTPUT_XLSX}")
