"""
print_predictions.py — Emit the assignment’s single-line CSV for “tomorrow”

What it does
------------
• Computes **tomorrow’s** 24 hourly load predictions per zone using
  `src.gam_predict.predict_10day_window(...)` with a 24h window anchored at
  tomorrow (local **US/Eastern**, EPT).
• Derives **tomorrow’s peak hour** per zone from those 24 values.
• Marks whether **tomorrow is a peak day** using either:
    (A) a **fixed** 10-day window passed via --fixed-start/--fixed-end, or
    (B) a **rolling** 10-day window [tomorrow .. tomorrow+9] (fallback).

Output format (to STDOUT)
-------------------------
A single CSV line:
  "YYYY-MM-DD",  ⟨24×|ZONES| hourly loads⟩,  ⟨|ZONES| peak hours⟩,  ⟨|ZONES| peak-day flags⟩

• Date is tomorrow’s date in ISO (quotes kept).
• Hourly loads are integers (rounded), ordered by `ZONES` and hour 00..23.
• Peak hour is the argmax hour (0..23) for tomorrow per zone.
• Peak-day flag is 1 if tomorrow is flagged as a peak day within the chosen
  10-day window, else 0.

Key details & assumptions
-------------------------
• Timezone: all timestamps are **EPT** (America/New_York / DST-aware).
• This script **does not write files** by itself; it prints to stdout so you
  can redirect in a Makefile, e.g.
      make predictions
      # internally:
      #   python -m src.print_predictions --fixed-start=YYYY-MM-DD --fixed-end=YYYY-MM-DD \
      #     > data/predictions_10day/one_line_$(date +%F).csv
• Zones order comes from `src.gam_predict.ZONES` (or auto-discovered if empty).

CLI
---
    python -m src.print_predictions \
        --fixed-start 2025-11-21 --fixed-end 2025-11-30

Optional flags
--------------
--fixed-start / --fixed-end : lock the peak-day window to a specific range.
--k                         : desired Top-K peak days (default 2).  Note:
                              in the current code path, the Top-K selection
                              is implemented inside `src.gam_predict` and this
                              flag is informational unless you wire it through.

Dependencies
------------
• Trained models & metadata under data/models/ (produced by src.train_gam_skl).
• Forecast inputs under data/raw/weather_forecast/ consumed by src.gam_predict.
"""


from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
import sys
import numpy as np
import pandas as pd

from src.gam_predict import predict_10day_window, ZONES

PRED_DIR = Path("data/predictions_10day"); PRED_DIR.mkdir(parents=True, exist_ok=True)
EASTERN = ZoneInfo("America/New_York")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-start", type=str, default=None,
                    help="YYYY-MM-DD start for fixed peak-day window")
    ap.add_argument("--fixed-end", type=str, default=None,
                    help="YYYY-MM-DD end for fixed peak-day window")
    ap.add_argument("--k", type=int, default=2, help="Top-K peak days (default 2)")
    args = ap.parse_args()

    # Tomorrow (local Eastern)
    target = (datetime.now(EASTERN).date() + timedelta(days=1))
    t_start = pd.Timestamp(target)            # 00:00 of tomorrow
    t_end   = pd.Timestamp(target)            # pass same day → 24h grid inside predict_10day_window()

    # 1) Get **tomorrow’s** 24 hourly loads & peak-hour (window = that single day)
    hourly_tomorrow, peaks_dummy = predict_10day_window(t_start.isoformat(), t_end.isoformat())
    if hourly_tomorrow.empty:
        print("[predictions] no hourly predictions for tomorrow.")
        return

    hourly_tomorrow["date"] = hourly_tomorrow["datetime_beginning_ept"].dt.date
    hourly_tomorrow["hour"] = hourly_tomorrow["datetime_beginning_ept"].dt.hour

    # Peak hour for tomorrow from those 24 values
    idx = hourly_tomorrow.groupby(["zone","date"])["mw_pred"].idxmax()
    peak_hour_tom = (hourly_tomorrow.loc[idx, ["zone","date","hour"]]
                                    .rename(columns={"hour":"peak_hour"}))

    # 2) Peak-day flag from a **fixed** window, if provided; else keep the rolling (tomorrow..+9) logic
    if args.fixed_start and args.fixed_end:
        fx_start = pd.Timestamp(args.fixed_start).normalize()
        fx_end   = pd.Timestamp(args.fixed_end).normalize()
        hourly_fixed, peaks_fixed = predict_10day_window(fx_start.isoformat(), fx_end.isoformat())
        if peaks_fixed.empty:
            # no flags available → default to 0
            peaks_fixed = pd.DataFrame(columns=["zone","date","is_peak_day"])
        # keep only the K largest days per zone (peaks_fixed already marks top-K in our implementation;
        # if not, recompute here by grouping day maxima and flagging nlargest(K))
        peak_flags = peaks_fixed.copy()
    else:
        # fallback: rolling 10-day window starting tomorrow (original assignment behavior)
        roll_start = t_start
        roll_end   = (t_start + pd.Timedelta(days=9))
        _, peaks_roll = predict_10day_window(roll_start.isoformat(), roll_end.isoformat())
        peak_flags = peaks_roll

    # Build the one-line CSV for the grader
    zones = ZONES if ZONES else sorted(hourly_tomorrow["zone"].unique())

    # 24 hourly loads for tomorrow per zone
    loads_by_zone = {}
    for z in zones:
        g = hourly_tomorrow[(hourly_tomorrow["zone"] == z) & (hourly_tomorrow["date"] == target)].sort_values("hour")
        s = g["mw_pred"].round().astype("Int64") if not g.empty else pd.Series([0]*24)
        if len(s) < 24:
            s = pd.concat([s, pd.Series([0]*(24-len(s)))], ignore_index=True)
        else:
            s = s.iloc[:24]
        loads_by_zone[z] = [int(x) for x in s.to_list()]

    # peak hour (tomorrow) per zone
    ph_by_zone = {}
    for z in zones:
        row = peak_hour_tom[(peak_hour_tom["zone"] == z) & (peak_hour_tom["date"] == target)]
        ph_by_zone[z] = int(row["peak_hour"].iloc[0]) if not row.empty else 0

    # peak-day flag from fixed/rolling window: 1 if tomorrow is flagged in that window (else 0)
    pd_by_zone = {}
    for z in zones:
        row = peak_flags[(peak_flags["zone"] == z) & (peak_flags["date"] == target)]
        pd_by_zone[z] = int(row["is_peak_day"].iloc[0]) if not row.empty else 0

    # Emit the required single line
    cells = [f"\"{target.isoformat()}\""]
    for z in zones: cells.extend(str(v) for v in loads_by_zone[z])  # 29×24
    for z in zones: cells.append(str(ph_by_zone.get(z, 0)))         # 29
    for z in zones: cells.append(str(pd_by_zone.get(z, 0)))         # 29
    print(", ".join(cells))

if __name__ == "__main__":
    main()