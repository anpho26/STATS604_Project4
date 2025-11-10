import pandas as pd
from pathlib import Path
from .constants import ZONES, MODEL_PATH, RAW_CSV

def main():
    raw = Path(RAW_CSV)
    if not raw.exists():
        print(f"[build] No raw data found at {raw}. Please add your CSV.")
        return

    df = pd.read_csv(raw)

    # Flexible timestamp column name
    tcol = "datetime" if "datetime" in df.columns else "datetime_beginning_ept"
    if tcol not in df.columns or "zone" not in df.columns or "mw" not in df.columns:
        raise ValueError("CSV needs columns: datetime/datetime_beginning_ept, zone, mw")

    # Parse, clean, keep only our 29 zones
    df["ts"] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=["ts", "mw"])
    df["hour"] = df["ts"].dt.hour
    df = df[df["zone"].isin(ZONES)]

    # Historical mean per (zone, hour)
    means = (
        df.groupby(["zone","hour"])["mw"]
          .mean()
          .rename("mean_mw")
          .reset_index()
    )

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    means.to_parquet(MODEL_PATH, index=False)
    print(f"[build] Saved hourly means -> {MODEL_PATH}")

if __name__ == "__main__":
    main()