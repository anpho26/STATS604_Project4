# src/gam_predict.py  (or src/models/gam_predict.py if you prefer)
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from joblib import load

# training utilities
from src.train_gam_skl import build_design

# Optional canonical ordering; leave empty to auto-discover from data
from src.constants import ZONES

MODEL_DIR = Path("data/models")
WX_FC_DIR = Path("data/raw/weather_forecast")
PRED_DIR  = Path("data/predictions_10day"); PRED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------

def _attach_temp_day_mean_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have ['zone','datetime_beginning_ept','temp'].
    Adds 'temp_day_mean' per zone/date.
    """
    out = df.copy()
    out["__date"] = out["datetime_beginning_ept"].dt.date
    out["temp_day_mean"] = out.groupby(["zone","__date"])["temp"].transform("mean")
    return out.drop(columns="__date")


def _list_zones_from_models() -> List[str]:
    zones = []
    for p in MODEL_DIR.glob("gam_skl_*.joblib"):
        z = p.stem.replace("gam_skl_", "")
        zones.append(z)
    return sorted(zones)

def _read_forecast_csvs_for_zone(zone: str) -> pd.DataFrame:
    cands = sorted(WX_FC_DIR.glob(f"*{zone}*.csv"))
    if not cands:
        return pd.DataFrame()
    parts = []
    for f in cands:
        df = pd.read_csv(f)

        # --- EPT only (rename if file uses 'time_ept') ---
        if "datetime_beginning_ept" not in df.columns:
            if "time_ept" in df.columns:
                df = df.rename(columns={"time_ept": "datetime_beginning_ept"})
            else:
                # if there is a UTC column, ignore it
                continue

        if "temp" not in df.columns and "temperature_2m" in df.columns:
            df = df.rename(columns={"temperature_2m": "temp"})
        if "temp" not in df.columns:
            continue

        if "zone" not in df.columns:
            df["zone"] = zone

        df["datetime_beginning_ept"] = pd.to_datetime(
            df["datetime_beginning_ept"], format="mixed", errors="coerce"
        )
        parts.append(df[["zone", "datetime_beginning_ept", "temp"]])

    if not parts:
        return pd.DataFrame()

    out = (pd.concat(parts, ignore_index=True)
             .dropna()
             .sort_values("datetime_beginning_ept"))
    # keep 1 row per (zone, timestamp)
    out = out.drop_duplicates(subset=["zone", "datetime_beginning_ept"], keep="last")
    return out

def _offset_from_meta(hours: pd.Series, hour_means_meta: Dict) -> pd.Series:
    """Map hour -> saved log1p(hour-mean). JSON may store keys as str."""
    def get(h):
        return hour_means_meta.get(h, hour_means_meta.get(str(h), np.nan))
    vals = hours.map(lambda h: float(get(int(h)))).astype(float)
    if np.isnan(vals).any():
        # fallback fill
        fill = float(np.nanmean([float(v) for v in hour_means_meta.values()]))
        vals = vals.fillna(fill if not np.isnan(fill) else 0.0)
    return vals


# ---------------- core prediction ----------------
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
        hours = pd.date_range(start_ts, end_excl, freq="h", inclusive="left")
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
        df["temp"] = df["temp"].fillna(daymean).ffill().bfill()

        # ---- prediction (log + offset) ----
        hrs = df["datetime_beginning_ept"].dt.hour
        off = _offset_from_meta(hrs, hour_means)

                # after you have df with columns ["zone","datetime_beginning_ept","temp"]
        df["date"] = df["datetime_beginning_ept"].dt.date
        df["temp_day_mean"] = df.groupby("date")["temp"].transform("mean")

        # (optional) if any day is fully NaN, fallback to ffill/bfill
        df["temp_day_mean"] = df["temp_day_mean"].ffill().bfill()

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

# ------------- compatibility for print_predictions -------------
def predict_all() -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, int]]:
    """
    Return (loads, peak_hour, peak_day_flag) for **tomorrow**:
      loads[z] -> 24-length np.array of MW
      peak_hour[z] -> int in [0..23] (tomorrow’s hour of max load)
      peak_day[z]  -> 0/1 whether tomorrow is one of the top-2 peak days (over 10-day window)
    """
    # define 10-day window starting tomorrow
    start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    end   = start + pd.Timedelta(days=9, hours=23)

    hourly, peaks = predict_10day_window(start.isoformat(), end.isoformat())
    if hourly.empty:
        return {}, {}, {}

    tomorrow = start.date()
    loads: Dict[str, np.ndarray] = {}
    ph: Dict[str, int] = {}
    pdflag: Dict[str, int] = {}

    zones = ZONES if ZONES else sorted(hourly["zone"].unique())
    for z in zones:
        g = hourly[(hourly["zone"] == z) & (hourly["date"] == tomorrow)].sort_values("hour")
        if len(g) == 24:
            loads[z] = g["mw_pred"].to_numpy(dtype=int)
            ph_row = peaks[(peaks["zone"] == z) & (peaks["date"] == tomorrow)]
            ph[z] = int(ph_row["peak_hour"].iloc[0]) if not ph_row.empty else int(g["mw_pred"].idxmax() % 24)
            pd_row = peaks[(peaks["zone"] == z) & (peaks["date"] == tomorrow)]
            pdflag[z] = int(pd_row["is_peak_day"].iloc[0]) if not pd_row.empty else 0
        else:
            # missing hours: fill with zeros to keep shape
            arr = np.zeros(24, dtype=int)
            if not g.empty:
                arr[g["hour"].to_numpy()] = g["mw_pred"].to_numpy(dtype=int)
            loads[z] = arr
            ph[z] = int(arr.argmax())
            pdflag[z] = 0

    return loads, ph, pdflag

# ------------- CLI (optional) -------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        h, p = predict_10day_window(sys.argv[1], sys.argv[2])
        if not h.empty:
            out_h = PRED_DIR / f"hourly_{sys.argv[1]}_{sys.argv[2]}.csv"
            out_p = PRED_DIR / f"peaks_{sys.argv[1]}_{sys.argv[2]}.csv"
            h.to_csv(out_h, index=False)
            p.to_csv(out_p, index=False)
            print(f"[GAM-skl] wrote {out_h} and {out_p}")
        else:
            print("[GAM-skl] no predictions produced.")
    else:
        # fallback: print tomorrow's one-line loads for quick smoke test
        loads, ph, pdflag = predict_all()
        print(f"[GAM-skl] zones={len(loads)}  sample={next(iter(loads)) if loads else '—'}")