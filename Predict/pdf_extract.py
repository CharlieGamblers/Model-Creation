# pdf_extract.py
import re
from pathlib import Path
import pandas as pd
import pdfplumber

def extract_bull_pens_from_pdf(pdf_path: Path, output_csv: Path = None, return_skipped=False):
    data = []
    skipped = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            pen_sections = re.split(r"Bull Pen\s+(\d+)", text)
            for i in range(1, len(pen_sections), 2):
                try:
                    pen_number = int(pen_sections[i])
                except ValueError:
                    continue

                pen_text = pen_sections[i + 1]
                matches = re.findall(r"([A-Z0-9\-]+)-(.+?)\s+([LR])\s+(.+)", pen_text)

                if not matches:
                    for line in pen_text.split("\n"):
                        if (
                            line.strip()
                            and not re.match(r"([A-Z0-9\-]+)-(.+?)\s+([LR])\s+(.+)", line.strip())
                        ):
                            skipped.append((pen_number, line.strip()))

                for stock_code, bull_number_and_name, delivery, contractor in matches:
                    parts = bull_number_and_name.strip().split(" ", 1)
                    bull_no = parts[0]
                    bull_name = parts[1] if len(parts) > 1 else ""

                    data.append({
                        "Pen": pen_number,
                        "Stock Code": stock_code.strip(),
                        "Bull No.": bull_no.strip(),
                        "Bull Name": bull_name.strip(),
                        "Delivery": delivery.strip(),
                        "Stock Contractor": contractor.strip(),
                        "Full Bull No.": f"{stock_code.strip()}-{bull_no.strip()}",
                    })

    df = pd.DataFrame(data)
    if output_csv:
        df.to_csv(output_csv, index=False)

    if return_skipped:
        return df, skipped
    return df
