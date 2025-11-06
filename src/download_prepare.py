# src/download_prepare.py
import io
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
RAW_TXT = DATA_DIR / "household_power_consumption.txt"
HOURLY_CSV = DATA_DIR / "energy_hourly.csv"

def download():
    if RAW_TXT.exists():
        print("Raw file already present.")
        return
    print("Downloading dataset...")
    with urllib.request.urlopen(URL) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        z.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")

def prepare():
    # Load; it's semicolon-separated with '?' as NA
    print("Preparing hourly dataset...")
    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values="?",
        low_memory=False
    )
    # Combine date/time and parse
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("datetime").set_index("datetime")

    # Convert 'Global_active_power' (kW) to kWh per hour by summing minute-level kW* (1/60 h)
    # Simpler: take mean kW for the hour * 1 hour -> approximate kWh
    hourly = df["Global_active_power"].astype(float).resample("1H").mean()  # kW
    hourly_kwh = hourly * 1.0  # 1 hour window → kWh approximation

    out = pd.DataFrame({"consumption_kwh": hourly_kwh})
    out = out.dropna().reset_index(names="timestamp")
    out.to_csv(HOURLY_CSV, index=False)
    print(f"Wrote {HOURLY_CSV} with {len(out)} rows.")

if __name__ == "__main__":
    download()
    prepare()