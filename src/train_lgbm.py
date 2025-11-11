# src/train_lgbm.py
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

DATA_DIR = Path("data")
POWER_DIR = DATA_DIR / "raw" / "power"
WEATH_DIR = DATA_DIR / "raw" / "weather"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ZONES = [
    "AECO","AEPAPT","AEPIMP","AEPKPT","AEPOPT","AP","BC","CE","DAY","DEOK",
    "DOM","DPLCO","DUQ","EASTON","EKPC","JC","ME","OE","OVEC","PAPWR","PE",
    "PEPCO","PLCO","PN","PS","RECO","SMECO","UGI","VMEU",
]

START_TRAIN = "2021-01-01"
END_TRAIN   = "2025-10-31"   # inclusive


TIME_FMT = "%m/%d/%Y %I:%M:%S %p"  # matches '1/1/2016 5:00:00 AM' etc.


def parse_pjm_time(s: pd.Series) -> pd.Series:
    # PJM strings look like '1/1/2016 12:00:00 AM'
    return pd.to_datetime(s, format=TIME_FMT, errors="coerce")

def load_power() -> pd.DataFrame:
    files = sorted(glob.glob(str(POWER_DIR / "hrl_load_metered_*.csv")))
    if not files:
        raise SystemExit(f"No PJM files found in {POWER_DIR}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, usecols=["load_area","datetime_beginning_ept","mw"])
        df = df.rename(columns={"load_area":"zone"})
        df["time_ept"] = pd.to_datetime(df["datetime_beginning_ept"], format=TIME_FMT, errors="coerce")
        df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
        frames.append(df[["zone","time_ept","mw"]])

    # ensure a fresh RangeIndex with no duplicates
    power = pd.concat(frames, ignore_index=True)

    # filter with a *positional* mask to avoid label alignment/reindex
    mask = power["zone"].isin(ZONES).to_numpy()
    power = power.loc[mask].dropna(subset=["time_ept","mw"])

    # keep training window and order
    power = power.loc[(power["time_ept"] >= START_TRAIN) & (power["time_ept"] <= END_TRAIN)]
    power = power.sort_values(["zone","time_ept"]).reset_index(drop=True)
    return power

def load_weather_zone(zone: str) -> pd.DataFrame:
    # concatenates all downloaded history files for this zone
    files = sorted(glob.glob(str(WEATH_DIR / f"meteostat_{zone}_*.csv")))
    if not files:
        raise SystemExit(f"No weather history files found for {zone} under {WEATH_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "datetime_beginning_ept" not in df.columns:
            raise SystemExit(f"{f} missing 'datetime_beginning_ept' (did you update weather_history.py?)")
        df["time_ept"] = parse_pjm_time(df["datetime_beginning_ept"])
        # Meteostat columns typically include: temp, rhum, prcp, wspd, pres, etc.
        keep = [c for c in ["temp","rhum","prcp","wspd","pres","coco"] if c in df.columns]
        df = df[["time_ept"] + keep]
        frames.append(df)
    wx = pd.concat(frames, ignore_index=True).dropna(subset=["time_ept"])
    wx = wx.sort_values("time_ept").drop_duplicates("time_ept", keep="last")
    return wx

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["time_ept"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek          # 0=Mon
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # day-of-year seasonality
    doy = dt.dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    # weather transforms (Celsius-based)
    if "temp" in df.columns:
        df["cdd"] = (df["temp"] - 18.0).clip(lower=0)
        df["hdd"] = (18.0 - df["temp"]).clip(lower=0)
    else:
        df["cdd"] = 0.0
        df["hdd"] = 0.0
    return df

def train_zone(zone: str, power: pd.DataFrame):
    df = power[power["zone"] == zone].copy()
    wx = load_weather_zone(zone)
    df = df.merge(wx, on="time_ept", how="left")
    df = add_features(df)

    # create safe lags using actuals
    df = df.sort_values("time_ept")
    df["lag24"]  = df["mw"].shift(24)
    df["lag168"] = df["mw"].shift(168)

    # drop early rows with missing lags or weather
    df = df.dropna(subset=["lag24","lag168"])

    target = "mw"
    features = [
        "hour","dow","month","is_weekend","doy_sin","doy_cos",
        "temp","rhum","prcp","wspd","pres","coco","cdd","hdd",
        "lag24","lag168"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    # simple model; you can tune later
    model = LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    # quick train RMSE (optional)
    pred = model.predict(X)
    # new
    mse = mean_squared_error(y, pred)
    rmse = float(np.sqrt(mse))
    print(f"[train] {zone}: n={len(df):,}, train RMSE={rmse:,.2f} MW")

    dump({"model": model, "features": features}, MODEL_DIR / f"lgbm_zone_{zone}.pkl")

def main():
    print("[train] loading power …")
    power = load_power()
    for z in ZONES:
        print(f"[train] zone {z}")
        train_zone(z, power)
    # save a small metadata file
    (MODEL_DIR / "lgbm_meta.json").write_text(json.dumps({
        "zones": ZONES, "start_train": START_TRAIN, "end_train": END_TRAIN
    }, indent=2))

if __name__ == "__main__":
    main()