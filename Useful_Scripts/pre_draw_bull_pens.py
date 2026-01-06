from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Predict import extract_bull_pens_from_pdf, get_top_lineups_with_insights
from Predict.config import replacements as _REPL_DF, DEFAULT_DATE

# Gamblers riders
gamblers_riders = [
    "Jose Vitor Leme",
    "Dalton Kasel",
    "Kaique Pacheco",
    "Dener Barbosa",
    "Sage Kimzey",
    "Lucas Divino",
    #"Andrei Scoparo",
    "Sage Kimzey",
    #"Vinicius Pinheiro Correa",
    "Callum Miller",
    "Ednelio Almeida",
    "Ramon de Lima"
]

def run_lineups_for_all_pens(pdf_path: Path, output_csv: Path, event_date: str, top_k: int = 10):
    # 1) Extract pens (+ skipped)
    pen_df, skipped = extract_bull_pens_from_pdf(pdf_path, output_csv, return_skipped=True)

    # 2) Build fast replacement dict: "8518 pneu dart's gold standard" -> "8518 Gold Standard"
    _repl = dict(zip(
        _REPL_DF["Known_Discrepancies"].str.lower().str.strip(),
        _REPL_DF["Replacements"].str.strip()
    ))

    def apply_replacement(bull_no: str, bull_name: str) -> tuple[str, str]:
        """
        Normalize and apply replacement mapping to "<bull_no> <bull_name>".
        Returns (fixed_no, fixed_name).
        """
        key = f"{str(bull_no).strip()} {str(bull_name).strip()}".lower()
        fixed = _repl.get(key)
        if not fixed:
            return str(bull_no).strip(), str(bull_name).strip()
        parts = fixed.split(" ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        # Fallback if replacement string is only a number/one token
        return parts[0].strip(), ""

    # 3) Iterate pens and run lineups using CLEANED bull strings
    for pen_num in sorted(pen_df["Pen"].unique()):
        pen_rows = pen_df.loc[pen_df["Pen"] == pen_num]

        # Apply replacements row-by-row, then build "NNNN Name" strings
        cleaned_bulls = []
        for _, r in pen_rows.iterrows():
            fixed_no, fixed_name = apply_replacement(r["Bull No."], r["Bull Name"])
            cleaned_bulls.append(f"{fixed_no} {fixed_name}".strip())

        # Cap at 5 bulls (your pen size)
        cleaned_bulls = cleaned_bulls[:5]

        print(f"\n=== Bull Pen {pen_num} ===")
        print(f"Bulls: {', '.join(cleaned_bulls) if cleaned_bulls else '[None]'}")

        # Show skipped (unparsed) lines for this pen
        skipped_here = [line for pen, line in skipped if pen == pen_num]
        if skipped_here:
            print(f"Skipped: {', '.join(skipped_here)}")

        # 4) Run lineups/insights on the cleaned list
        get_top_lineups_with_insights(
            riders=gamblers_riders,
            bulls=cleaned_bulls,
            event_date=event_date,
            k=top_k
        )

if __name__ == "__main__":
    PDF_PATH = Path(r"C:\Users\CharlieCampbell\OneDrive - Austin Gamblers\Documents\Day 1 Bull Pens Pre Drawn.pdf")
    OUT_CSV = Path("Data/Processed/bull_pens.csv")
    EVENT_DATE = DEFAULT_DATE
    TOP_K = 5

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    run_lineups_for_all_pens(
        pdf_path=PDF_PATH,
        output_csv=OUT_CSV,
        event_date=EVENT_DATE,
        top_k=TOP_K,
    )
