from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sys
import pandas as pd

# from src.forecast10days import main as fc_main
from src.gam_predict import predict_10day_window, ZONES

# call existing modules programmatically
from src.downloads.weather_forecast import main as wx_main

PRED_DIR = Path("data/predictions_10day")
EASTERN = ZoneInfo("America/New_York")


def ensure_window(start: str, end: str) -> tuple[Path, Path]:
    """Make sure hourly_ and peaks_ CSVs for [start,end] exist; build if not."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    hourly = PRED_DIR / f"hourly_{start}_{end}.csv"
    peaks  = PRED_DIR / f"peaks_{start}_{end}.csv"
    if not (hourly.exists() and peaks.exists()):
        # 1) fetch forecast for all zones
        sys.argv = ["weather_forecast", start, end, "--all"]
        wx_main()

        # 2) run GAM predictor and write CSVs in the format this script expects
        hourly_df, peaks_df = predict_10day_window(start, end)
        # normalize column names for backward compatibility
        hourly_df = hourly_df.rename(columns={"datetime_beginning_ept": "time_ept"})
        peaks_df  = peaks_df.rename(columns={"is_peak_day": "peak_day_flag"})
        hourly_df.to_csv(hourly, index=False)
        peaks_df.to_csv(peaks, index=False)
    return hourly, peaks


def main():
    # Tomorrow in US/Eastern for the header date and for selecting rows
    target = (datetime.now(EASTERN).date() + timedelta(days=1))
    start  = target.isoformat()
    end    = (target + timedelta(days=9)).isoformat()

    hourly_path, peaks_path = ensure_window(start, end)

    # Load predictions
    hourly = pd.read_csv(hourly_path, parse_dates=["time_ept"])
    peaks  = pd.read_csv(peaks_path)
    # Normalize date typing
    if "date" in peaks.columns:
        peaks["date"] = pd.to_datetime(peaks["date"]).dt.date

    # Filter to the target day
    day = hourly[hourly["time_ept"].dt.date == target].copy()

    # Build dictionaries for 24 hourly loads, peak hour, peak-day flag
    loads_by_zone: dict[str, list[int]] = {}
    ph_by_zone: dict[str, int] = {}
    pd_by_zone: dict[str, int] = {}

    # peaks file: one row per zone/date with peak_hour and peak_day_flag
    peaks_t = peaks[peaks["date"] == target]
    if not peaks_t.empty:
        ph_by_zone.update({r["zone"]: int(r["peak_hour"]) for _, r in peaks_t.iterrows()})
        pd_by_zone.update({r["zone"]: int(r["peak_day_flag"]) for _, r in peaks_t.iterrows()})

    for z in ZONES:
        d = day[day["zone"] == z].sort_values("time_ept")
        # 24 hourly loads; round to nearest int
        s = d["mw_pred"].round().astype("Int64") if not d.empty else pd.Series([0]*24)
        # If somehow fewer than 24 hours, pad with zeros
        if len(s) < 24:
            s = pd.concat([s, pd.Series([0]*(24-len(s)))], ignore_index=True)
        else:
            s = s.iloc[:24]
        loads_by_zone[z] = [int(x) for x in s.to_list()]

        # Fallback peak hour if missing: argmax of the 24h loads
        if z not in ph_by_zone and not d.empty:
            ph_by_zone[z] = int(d.loc[d["mw_pred"].idxmax(), "time_ept"].hour)
        if z not in pd_by_zone:
            pd_by_zone[z] = 0

    # Assemble required one-line CSV
    cells = [f"\"{target.isoformat()}\""]  # "YYYY-MM-DD"
    # L1_00..L1_23, L2_00.., ..., L29_23
    for z in ZONES:
        cells.extend(str(v) for v in loads_by_zone[z])
    # PH_1..PH_29
    for z in ZONES:
        cells.append(str(ph_by_zone.get(z, 0)))
    # PD_1..PD_29
    for z in ZONES:
        cells.append(str(pd_by_zone.get(z, 0)))

    print(", ".join(cells))


if __name__ == "__main__":
    main()