# src/forecast10days_gam_skl.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json, numpy as np, pandas as pd
from joblib import load

# Optional canonical ordering; leave empty to auto-discover from data
from src.constants import ZONES

MODEL_DIR = Path("data/models")
WX_FC_DIR = Path("data/raw/weather_forecast")
OUTDIR    = Path("data/predictions_10day"); OUTDIR.mkdir(parents=True, exist_ok=True)

from src.train_gam_skl import build_design  # same feature map

def load_zone_forecast_temp(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Expect CSVs in data/raw/weather_forecast with columns: zone, time_ept, temp (or temperature_2m)
    dfs=[]
    for f in sorted(WX_FC_DIR.glob(f"*{zone}*.csv")):
        df = pd.read_csv(f)
        if "zone" not in df.columns: df["zone"] = zone
        if "time_ept" not in df.columns:
            for c in df.columns:
                if c.lower().startswith("time_"):
                    df = df.rename(columns={c:"time_ept"}); break
        if "temp" not in df.columns and "temperature_2m" in df.columns:
            df = df.rename(columns={"temperature_2m":"temp"})
        df["time_ept"] = pd.to_datetime(df["time_ept"], errors="coerce")
        dfs.append(df[["zone","time_ept","temp"]])
    if not dfs:
        raise FileNotFoundError(f"No forecast temp files for {zone}")
    df = pd.concat(dfs, ignore_index=True)
    m = (df["zone"]==zone) & (df["time_ept"]>=start) & (df["time_ept"]<=end)
    return df.loc[m].dropna().sort_values("time_ept")

def predict_10day_window(start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ---- 1) normalize window to [start, end+1d) ----
    start_ts = pd.to_datetime(start).normalize()
    end_excl = pd.to_datetime(end).normalize() + pd.Timedelta(days=1)

    zones = ZONES if ZONES else _list_zones_from_models()
    hourly_rows: List[pd.DataFrame] = []

    for z in zones:
        model_path = MODEL_DIR / f"gam_skl_{z}.joblib"
        meta_path  = MODEL_DIR / f"gam_skl_{z}.meta.json"
        if not model_path.exists() or not meta_path.exists():
            print(f"[GAM-skl] missing model/meta for {z}, skipping")
            continue

        pipe = load(model_path)
        meta = json.loads(meta_path.read_text())
        hour_means = meta.get("hour_means_log1p", {})
        delta      = float(meta.get("delta", 0.0))

        # ---- 2) full grid of hours (authoritative) ----
        hours = pd.date_range(start_ts, end_excl, freq="H", inclusive="left")
        grid  = pd.DataFrame({"zone": z, "datetime_beginning_ept": hours})

        # ---- 3) load forecast temps and merge onto grid ----
        fc = _read_forecast_csvs_for_zone(z)
        if fc.empty:
            print(f"[GAM-skl] empty forecast CSVs for {z}")
            continue

        fc = (fc[(fc["datetime_beginning_ept"] >= start_ts) &
                 (fc["datetime_beginning_ept"] <  end_excl)]
                .drop_duplicates(subset=["datetime_beginning_ept"], keep="last"))

        df = grid.merge(fc[["datetime_beginning_ept","temp"]], on="datetime_beginning_ept", how="left")

        # Fill temp by day-mean, then ffill/bfill as last resort
        df["date"] = df["datetime_beginning_ept"].dt.date
        daymean = df.groupby("date")["temp"].transform("mean")
        df["temp"] = df["temp"].fillna(daymean).fillna(method="ffill").fillna(method="bfill")

        # ---- prediction (log + offset) ----
        hrs = df["datetime_beginning_ept"].dt.hour
        off = _offset_from_meta(hrs, hour_means)

        X, _ = build_design(df)
        yhat_log = pipe.predict(X.values) + off.values + delta
        yhat = np.expm1(yhat_log)

        hourly_rows.append(pd.DataFrame({
            "zone": z,
            "datetime_beginning_ept": df["datetime_beginning_ept"].values,
            "mw_pred": np.round(yhat).astype(int)
        }))

    if not hourly_rows:
        return pd.DataFrame(), pd.DataFrame()

    hourly = (pd.concat(hourly_rows, ignore_index=True)
                .sort_values(["zone","datetime_beginning_ept"]))
    hourly["date"] = hourly["datetime_beginning_ept"].dt.date
    hourly["hour"] = hourly["datetime_beginning_ept"].dt.hour

    # ---- 4) sanity: every zone/day must have 24 hours ----
    bad = []
    for (z, d), g in hourly.groupby(["zone", "date"]):
        if len(g) != 24:
            missing = sorted(set(range(24)) - set(g["hour"]))
            bad.append((z, str(d), missing))
    if bad:
        print("[GAM-skl][WARN] Incomplete days (zone, date, missing_hours):")
        for z, d, miss in bad[:12]:
            print(f"  {z} {d}  missing={miss}")
        if len(bad) > 12:
            print(f"  … and {len(bad)-12} more")

    # peak hour per zone/day
    idx = hourly.groupby(["zone","date"])["mw_pred"].idxmax()
    peak_hour = hourly.loc[idx, ["zone","date","hour"]].rename(columns={"hour":"peak_hour"})

    # top-2 peak-day flag
    daymax = hourly.groupby(["zone","date"])["mw_pred"].max()
    flags = []
    for z, g in daymax.groupby(level=0):
        f = pd.Series(0, index=g.index, dtype=int)
        f.loc[g.nlargest(2).index] = 1
        flags.append(f.rename("is_peak_day").reset_index())
    peaks = peak_hour.merge(pd.concat(flags, ignore_index=True), on=["zone","date"])
    return hourly, peaks

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.forecast10days_gam_skl START END (YYYY-MM-DD)"); raise SystemExit(2)
    hourly, peaks = predict_10day_window(sys.argv[1], sys.argv[2])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hp = OUTDIR / f"hourly_{sys.argv[1]}_{sys.argv[2]}.csv"
    pp = OUTDIR / f"peaks_{sys.argv[1]}_{sys.argv[2]}.csv"
    hourly.to_csv(hp, index=False)
    peaks.to_csv(pp, index=False)
    print(f"[GAM-skl] wrote {hp}")
    print(f"[GAM-skl] wrote {pp}")

if __name__ == "__main__":
    main()