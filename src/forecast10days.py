# src/forecast_10day.py
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from joblib import load

DATA_DIR = Path("data")
PRED_DIR = DATA_DIR / "predictions_10day"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = DATA_DIR / "models"
POWER_DIR = DATA_DIR / "raw" / "power"
WXF_DIR   = DATA_DIR / "raw" / "weather_forecast"

ZONES = [
    "AECO","AEPAPT","AEPIMP","AEPKPT","AEPOPT","AP","BC","CE","DAY","DEOK",
    "DOM","DPLCO","DUQ","EASTON","EKPC","JC","ME","OE","OVEC","PAPWR","PE",
    "PEPCO","PLCO","PN","PS","RECO","SMECO","UGI","VMEU",
]

TIME_FMT = "%m/%d/%Y %I:%M:%S %p"  # matches '1/1/2016 5:00:00 AM' etc.

def parse_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format=TIME_FMT, errors="coerce")

def load_last_history(zone: str, until_ept: pd.Timestamp) -> pd.DataFrame:
    """Return at least 168+24 hours of actuals before 'until_ept'."""
    import glob
    files = sorted(glob.glob(str(POWER_DIR / "hrl_load_metered_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["load_area","datetime_beginning_ept","mw"])
        df = df.rename(columns={"load_area":"zone"})
        df["time_ept"] = parse_time(df["datetime_beginning_ept"])
        frames.append(df)
    hist = pd.concat(frames, ignore_index=True)
    hist = hist[(hist["zone"]==zone) & (hist["time_ept"] < until_ept)].sort_values("time_ept")
    # keep only last 4000 hours to reduce memory
    hist = hist.iloc[-4000:]
    return hist[["time_ept","mw"]].reset_index(drop=True)

def build_features(times: pd.DatetimeIndex, wx: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"time_ept": times})
    dt = df["time_ept"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    doy = dt.dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    # merge weather forecast (which already has PJM-style columns)
    wx = wx.rename(columns={"datetime_beginning_ept":"time_ept"})
    df = df.merge(wx, on="time_ept", how="left")
    # weather transforms
    if "temp" in df.columns:
        df["cdd"] = (df["temp"] - 18.0).clip(lower=0)
        df["hdd"] = (18.0 - df["temp"]).clip(lower=0)
    else:
        df["cdd"] = 0.0
        df["hdd"] = 0.0
    return df

def load_forecast_zone(zone: str, start: str, end: str) -> pd.DataFrame:
    f = WXF_DIR / f"openmeteo_forecast_{zone}_{start}_{end}.csv"
    df = pd.read_csv(f)
    df["datetime_beginning_ept"] = parse_time(df["datetime_beginning_ept"])

    # allow both Open-Meteo names and already-normalized names
    rename = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "rhum",
        "precipitation": "prcp",
        "windspeed_10m": "wspd",
        "pressure_msl": "pres",
        "weathercode": "coco",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

def predict_zone(zone: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_pack = load(MODEL_DIR / f"lgbm_zone_{zone}.pkl")
    model, feats = model_pack["model"], model_pack["features"]

    start_ts = pd.Timestamp(start)  # inclusive
    end_ts   = pd.Timestamp(end)    # inclusive end-date; we'll stop at end 23:00
    times = pd.date_range(start_ts, end_ts + pd.Timedelta(hours=23), freq="h")

    # 1) history for lags
    hist = load_last_history(zone, until_ept=start_ts)
    if len(hist) < 200:
        raise SystemExit(f"Not enough history for {zone}")

    # 2) forecast weather
    wxf = load_forecast_zone(zone, start, end)

    # 3) build static features for all future hours
    Xall = build_features(times, wxf)

    # 4) roll forward hour-by-hour to fill lag24/lag168 and predict
    series = hist.copy()
    preds = []
    for t in times:
        row = Xall.loc[Xall["time_ept"] == t].copy()
        # compute lags from (actual history + previous predictions)
        val_24  = series.loc[series["time_ept"] == t - pd.Timedelta(hours=24), "mw"]
        val_168 = series.loc[series["time_ept"] == t - pd.Timedelta(hours=168), "mw"]
        row["lag24"]  = float(val_24.iloc[0])  if len(val_24)  else np.nan
        row["lag168"] = float(val_168.iloc[0]) if len(val_168) else np.nan
        # if an early hour somehow lacks a lag (shouldn't), backoff to last known
        if np.isnan(row["lag24"].values[0]) or np.isnan(row["lag168"].values[0]):
            row["lag24"] = series["mw"].iloc[-24:].mean()
            row["lag168"] = series["mw"].iloc[-168:].mean()
        # keep only model features
        x = row[feats]
        yhat = float(model.predict(x)[0])
        preds.append({"zone": zone, "time_ept": t, "mw_pred": yhat})
        series = pd.concat([series, pd.DataFrame({"time_ept":[t], "mw":[yhat]})], ignore_index=True)

    pred_df = pd.DataFrame(preds)

    # daily peaks (no groupby.apply warning)
    pred_df["date"] = pred_df["time_ept"].dt.date
    idx = pred_df.groupby(["zone", "date"])["mw_pred"].idxmax()
    day_max = (pred_df.loc[idx, ["zone", "date", "time_ept", "mw_pred"]]
            .rename(columns={"mw_pred": "peak_load"}))
    day_max["peak_hour"] = day_max["time_ept"].dt.hour
    day_max = day_max.drop(columns=["time_ept"])

    # top-2 peak days per zone
    day_max["rank"] = day_max.groupby("zone")["peak_load"].rank(method="first", ascending=False)
    day_max["peak_day_flag"] = (day_max["rank"] <= 2).astype(int)
    day_max = day_max.drop(columns="rank")

    return pred_df, day_max

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.forecast_10day START END")
        print("Example: python -m src.forecast_10day 2025-11-20 2025-11-29")
        sys.exit(2)
    start, end = sys.argv[1], sys.argv[2]

    all_h = []
    all_p = []
    for z in ZONES:
        print(f"[forecast] {z}")
        h, p = predict_zone(z, start, end)
        all_h.append(h); all_p.append(p)

    hourly = pd.concat(all_h, ignore_index=True)
    peaks  = pd.concat(all_p, ignore_index=True)

    # save
    hourly_out = PRED_DIR / f"hourly_{start}_{end}.csv"
    peaks_out  = PRED_DIR / f"peaks_{start}_{end}.csv"
    hourly.to_csv(hourly_out, index=False)
    peaks.to_csv(peaks_out, index=False)
    print(f"[forecast] wrote {hourly_out}")
    print(f"[forecast] wrote {peaks_out}")

if __name__ == "__main__":
    main()