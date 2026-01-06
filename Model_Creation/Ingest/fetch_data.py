# fetch_data.py

import requests
import pandas as pd
import json
import os
from pathlib import Path

PBR_KEY = "ea27df0c9387122daefcbcb4c165bedf4d01c3d5"
BASE_URL = "https://probullstats.com/api/api.php"

def fetch_rider_hands():
    url = f"{BASE_URL}?key={PBR_KEY}&data=rider_hands"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def extract_rider_hands(data):
    hands = []
    for entry in data:
        try:
            hands.append({
                "rider": entry["rider"].lower().strip(),
                "hand": entry["hand"].strip(),
                "rider_id": int(entry["rider_id"])
            })
        except (KeyError, TypeError, ValueError):
            print(f"Skipping malformed hand entry: {entry}")
            continue
    return pd.DataFrame(hands)


def get_highest_row_id():
    url = f"{BASE_URL}?data=fcmax&key={PBR_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and len(data) == 1:
        return 583300
    raise ValueError("Unexpected response format for row ID")

def fetch_outs_data(start_rowid, batch_size=20000, delay=2.5, max_batches=15):
    import time
    current_rowid = start_rowid
    all_data = []
    for _ in range(max_batches):
        url = f"{BASE_URL}?data=outs_set_a&id={current_rowid}&key={PBR_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            batch = response.json()
            all_data.extend(batch)
            print(f"Fetched batch starting from row ID: {current_rowid}")
            current_rowid -= batch_size
            time.sleep(delay)
        else:
            print(f"Failed at row ID: {current_rowid} - Status {response.status_code}")
            break
    return pd.json_normalize(all_data)

if __name__ == "__main__":
    CUR = Path(__file__).resolve()
    for p in CUR.parents:
        if p.name == "Bull_Model":
            ROOT = p
            break
    else:
        ROOT = CUR.parents[2]

    (ROOT / "Data" / "Raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "Data" / "Processed").mkdir(parents=True, exist_ok=True)

    print("Fetching rider hands...")
    rider_data = fetch_rider_hands()
    rider_df = extract_rider_hands(rider_data)
    rider_df.to_csv(ROOT / "Data" / "Processed" / "rider_info.csv", index=False)

    print("Fetching highest row ID for outs...")
    highest_id = get_highest_row_id() + 1
    print(f"Starting from: {highest_id}")

    outs_df = fetch_outs_data(highest_id)
    outs_df.to_csv(ROOT / "Data" / "Raw" / "base_data.csv", index=False)
    print("Saved all fetched data.")
