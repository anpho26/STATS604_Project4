"""
GAM-skl 10-day load forecaster (hourly predictions + peak-day flags).

Purpose
-------
Produce hourly load forecasts for a fixed calendar window (typically the 10-day
Thanksgiving window) for all zones with trained models, and mark candidate
"peak" days. Can be used as a library (returns DataFrames) or as a CLI that
also writes CSVs under `data/predictions_10day/`.

Dependencies / Inputs
---------------------
1) Trained per-zone models and metadata (created by `src.train_gam_skl`):
   - `data/models/gam_skl_{ZONE}.joblib`
   - `data/models/gam_skl_{ZONE}.meta.json`
     with keys:
       * `hour_means_log1p` : dict[int -> float]  (same-hour log baseline)
       * `delta`            : float               (last-14-days bias tweak)

2) Weather forecast CSVs per zone (created by `src/downloads/weather_forecast.py`)
   in `data/raw/weather_forecast/`, containing at least:
   - `datetime_beginning_ept`  (or `time_ept`; UTC is ignored)
   - `temperature_2m` (or `temp`)
   - optional `zone` (filled from filename if missing)

Outputs (library)
-----------------
`predict_10day_window(start, end) -> (hourly_df, peaks_df)`

- `hourly_df` columns:
    ['zone', 'datetime_beginning_ept', 'mw_pred', 'date', 'hour']
- `peaks_df` columns:
    ['zone', 'date', 'peak_hour', 'is_peak_day']

Notes on behavior
-----------------
- The window is normalized to **[start, end + 1 day)** in EPT and an
  *authoritative hourly grid* is built, ensuring 24 rows per zone×day.
- Missing forecast temps are imputed by same-day mean, then ffill/bfill.
- Design matrix comes from `src.train_gam_skl.build_design` and includes the
  daily mean temperature feature that was used to train.
- Predictions are produced in log-space with the saved hour offset + bias delta,
  then exponentiated back to MW.

Peak-day logic
--------------
- For each zone we compute the daily max `mw_pred`. We then flag up to TOP_K
  (currently 3) largest days **excluding Thanksgiving and Black Friday** for
  that year. (If exclusion removes all candidates, we fall back to the largest
  days without exclusion.)
- `is_peak_day` is 1 for flagged days, else 0.
- `peak_hour` is the argmax hour within each zone×day (0..23).

Convenience
-----------
`predict_all()` builds the [tomorrow .. +9 days] window from "now" and returns:
    (loads, peak_hour, peak_day_flag)
where `loads[z]` is a 24-length int array for tomorrow.

CLI usage
---------
From the repo root:

    # run as a script and write CSVs
    python -m src.gam_predict 2025-11-21 2025-11-30

This writes:
    data/predictions_10day/hourly_2025-11-21_2025-11-30.csv
    data/predictions_10day/peaks_2025-11-21_2025-11-30.csv

Programmatic usage
------------------
    from src.gam_predict import predict_10day_window
    hourly, peaks = predict_10day_window("2025-11-21", "2025-11-30")

Assumptions & pitfalls
----------------------
- All times are **EPT**. If your forecast files only have UTC, convert upstream.
- Models must exist for the zones you want; otherwise the zone is skipped with
  a log message.
- If any zone×day yields fewer than 24 rows after merging, we print a warning
  listing the missing hours.
"""


from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from joblib import load
from datetime import date, timedelta

# training utilities
from src.train_gam_skl import build_design

# Optional canonical ordering; leave empty to auto-discover from data
from src.constants import ZONES

MODEL_DIR = Path("data/models")
WX_FC_DIR = Path("data/raw/weather_forecast")
PRED_DIR  = Path("data/predictions_10day"); PRED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def thanksgiving_date(year: int) -> date:
    """4th Thursday in November."""
    d = date(year, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(weeks=3)

def _is_tg_or_bf(d: date) -> bool:
    """Return True if d is Thanksgiving or Black Friday for its year."""
    tg = thanksgiving_date(d.year)
    return (d == tg) or (d == tg + timedelta(days=1))

def choose_peak_days_for_zone(
    s: pd.Series,
    k_min: int = 2,
    k_max: int = 4,
    ratio: float = 0.98,
    exclude_thanksgiving: bool = True,
) -> set:
    """
    s: Series indexed by python `date` with daily max load for ONE zone.
    - Always pick at least k_min days.
    - Optionally add more days nearly tied with #2 (within `ratio`), capped at k_max.
    - If exclude_thanksgiving=True, never flag Thanksgiving.
    """
    s_sorted = s.sort_values(ascending=False)

    # Build exclusion set
    ex = set()
    if exclude_thanksgiving:
        ex = {thanksgiving_date(d.year) for d in s.index}

    keep: list[date] = []

    # 1) ensure k_min non-excluded days
    for d, v in s_sorted.items():
        if d in ex: 
            continue
        keep.append(d)
        if len(keep) == k_min:
            break

    # In the very unlikely case we still don't have k_min (shouldn’t happen in a 10-day window),
    # continue picking next best non-excluded days:
    if len(keep) < k_min:
        for d, v in s_sorted.items():
            if d in keep or d in ex:
                continue
            keep.append(d)
            if len(keep) == k_min:
                break

    # 2) near-ties w.r.t. the second kept value
    m2 = float(s[keep[min(1, len(keep)-1)]])  # second kept (or first if only one)
    for d, v in s_sorted.items():
        if d in keep or d in ex:
            continue
        if len(keep) >= k_max:
            break
        if v >= ratio * m2:
            keep.append(d)
        else:
            break

    return set(keep)

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

    # --- peak-day flags (exclude Thanksgiving & Black Friday) ---
    daymax = hourly.groupby(["zone","date"])["mw_pred"].max()
    flags = []
    TOP_K = 3   # we’re allowed to guess >2; pick up to 3 best non-TG/BF days

    for z, g in daymax.groupby(level=0):
        df = g.reset_index()                    # columns: ['zone','date','mw_pred'] (name of g is 'mw_pred')
        valcol = g.name if g.name in df.columns else df.columns[-1]
        # keep only non-holiday candidates
        keep = df[~df["date"].map(_is_tg_or_bf)]
        if keep.empty:                          # fallback: if ALL 10 days are TG/BF (unlikely), don't exclude
            keep = df
        chosen_idx = (keep.nlargest(TOP_K, valcol)
                        .set_index(["zone","date"])
                        .index)
        f = pd.Series(0, index=g.index, dtype=int)
        f.loc[chosen_idx] = 1
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