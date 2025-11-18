from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Reuse your training/pred utils
from src.train_gam_skl import (
    load_power, load_temp_history, build_design,
    tg_window, thanksgiving_date
)
from joblib import load as joblib_load

# Optional: direct call into your predictor (for the 2025 window)
from src.gam_predict import predict_10day_window

MODEL_DIR = Path("data/models")
WX_HIST_DIR = Path("data/raw/weather")
FIG_DIR = Path("data/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)


def _offset_from_meta(hours: pd.Series, hour_means_meta: Dict) -> pd.Series:
    """Map hour -> saved log1p(hour mean). JSON may store keys as str."""
    def get(h):
        return hour_means_meta.get(h, hour_means_meta.get(str(h), np.nan))
    vals = hours.map(lambda h: float(get(int(h)))).astype(float)
    if np.isnan(vals).any():
        fill = float(np.nanmean([float(v) for v in hour_means_meta.values()]))
        vals = vals.fillna(fill if not np.isnan(fill) else 0.0)
    return vals


def _attach_temp_day_mean_hist(df_wx: pd.DataFrame) -> pd.DataFrame:
    """Given weather history with hourly 'temp', attach per-day mean temp."""
    out = df_wx.copy()
    out["__date"] = out["datetime_beginning_ept"].dt.date
    out["temp_day_mean"] = out.groupby(["zone", "__date"])["temp"].transform("mean")
    return out.drop(columns="__date")


def _read_2024_backtest_frame(zone: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (design_df_for_2024_window, actual_power_for_window) for a zone,
    using Meteostat history (EPT) for temps.
    """
    start24, end24 = tg_window(2024)
    power = load_power()
    wx = load_temp_history()

    # filter to window/zone
    pz = (power[(power["zone"] == zone) &
                (power["datetime_beginning_ept"] >= start24) &
                (power["datetime_beginning_ept"] <= end24)]
          .copy())
    wz = (wx[(wx["zone"] == zone) &
             (wx["datetime_beginning_ept"] >= start24) &
             (wx["datetime_beginning_ept"] <= end24)]
          .copy())

    # authoritative hourly grid to avoid gaps
    hours = pd.date_range(start24, end24 + pd.Timedelta(hours=23), freq="h")
    grid = pd.DataFrame({"zone": zone, "datetime_beginning_ept": hours})

    # merge temp onto grid; fill temp by within-day mean then ffill/bfill
    df = grid.merge(wz[["datetime_beginning_ept", "temp"]], on="datetime_beginning_ept", how="left")
    df["date"] = df["datetime_beginning_ept"].dt.date
    daymean = df.groupby("date")["temp"].transform("mean")
    df["temp"] = df["temp"].fillna(daymean).ffill().bfill()

    # attach daily mean temp feature (what your model uses)
    df["temp_day_mean"] = df.groupby("date")["temp"].transform("mean")
    df["temp_day_mean"] = df["temp_day_mean"].ffill().bfill()

    return df.drop(columns=["date"]), pz.sort_values("datetime_beginning_ept")


def plot_2024_actual_vs_pred(zone: str) -> Path:
    """Plot actual vs prediction for 2024 TG window (hourly)."""
    start24, end24 = tg_window(2024)

    # load 2025-trained model (pipeline) and meta
    model_path = MODEL_DIR / f"gam_skl_{zone}.joblib"
    meta_path  = MODEL_DIR / f"gam_skl_{zone}.meta.json"
    if not (model_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Missing model/meta for {zone}")

    pipe = joblib_load(model_path)
    meta = json.loads(meta_path.read_text())
    hour_means = meta["hour_means_log1p"]
    delta = float(meta.get("delta", 0.0))

    df, pz = _read_2024_backtest_frame(zone)

    # features + offset
    X, _ = build_design(df)
    off = _offset_from_meta(df["datetime_beginning_ept"].dt.hour, hour_means)
    yhat = np.expm1(pipe.predict(X.values) + off.values + delta)

    # align to actual
    pred = pd.DataFrame({
        "datetime_beginning_ept": df["datetime_beginning_ept"].values,
        "mw_pred": yhat
    })
    merged = pz[["datetime_beginning_ept","mw"]].merge(
        pred, on="datetime_beginning_ept", how="inner"
    ).sort_values("datetime_beginning_ept")

    # metrics
    rmse = float(np.sqrt(np.mean((merged["mw"] - merged["mw_pred"])**2)))
    # peak hour ±1 accuracy
    gt_peak_idx = merged.groupby(merged["datetime_beginning_ept"].dt.date)["mw"].idxmax()
    pr_peak_idx = merged.groupby(merged["datetime_beginning_ept"].dt.date)["mw_pred"].idxmax()
    gt_hours = merged.loc[gt_peak_idx, "datetime_beginning_ept"].dt.hour.values
    pr_hours = merged.loc[pr_peak_idx, "datetime_beginning_ept"].dt.hour.values
    acc_pm1 = float(np.mean(np.abs(gt_hours - pr_hours) <= 1))

    # plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(merged["datetime_beginning_ept"], merged["mw"], label="Actual 2024 (MW)")
    ax.plot(merged["datetime_beginning_ept"], merged["mw_pred"], label="Predicted 2024 (MW)")
    ax.set_title(f"{zone} — 2024 TG window  (RMSE={rmse:.1f}, Peak ±1 acc={acc_pm1:.2f})")
    ax.set_xlabel("Time (EPT)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))

    # shade Thanksgiving & Black Friday
    tg = thanksgiving_date(2024)
    bf = tg + timedelta(days=1)
    ax.axvspan(pd.Timestamp(tg), pd.Timestamp(tg) + pd.Timedelta(days=1), alpha=0.1)
    ax.axvspan(pd.Timestamp(bf), pd.Timestamp(bf) + pd.Timedelta(days=1), alpha=0.1)

    out = FIG_DIR / f"{zone}_2024_actual_vs_pred.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_2025_pred_vs_past(zone: str, years: List[int] = [2022, 2023, 2024]) -> Path:
    """
    Plot: 2025 prediction vs past *actuals* for the same TG 10-day window.
    Assumes forecast CSVs exist for the 2025 window.
    """
    start25, end25 = tg_window(2025)
    # 2025 prediction (hourly)
    hourly_25, _ = predict_10day_window(start25.isoformat(), end25.isoformat())
    g25 = (hourly_25[(hourly_25["zone"] == zone)]
           .sort_values("datetime_beginning_ept")
           .rename(columns={"mw_pred":"MW"}))

    # Past years’ actuals on their TG windows
    power = load_power()

    # Build a time axis aligned to 2025 window (hours 0..239)
    # and plot each series against that index for clean comparisons
    g25 = g25.assign(tidx=np.arange(len(g25)))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(g25["tidx"], g25["MW"], linewidth=2.0, label="2025 prediction")

   # in plot_2025_pred_vs_past(...)

    for y in years:
        s, e = tg_window(y)

        # canonical 240h grid for the year's TG window
        hours = pd.date_range(s, e + pd.Timedelta(days=1), freq="h", inclusive="left")
        grid = pd.DataFrame({"datetime_beginning_ept": hours})

        gy_raw = power[(power["zone"] == zone)
                    & (power["datetime_beginning_ept"] >= s)
                    & (power["datetime_beginning_ept"] <  e + pd.Timedelta(days=1))]
        gy = (grid.merge(
                gy_raw[["datetime_beginning_ept", "mw"]],
                on="datetime_beginning_ept", how="left")
            .sort_values("datetime_beginning_ept"))

        # Option A (recommended): leave NaNs → plot shows gaps if data is missing
        y_series = gy["mw"].to_numpy()

        # Option B (smooth fill): uncomment to avoid gaps
        # y_series = gy["mw"].interpolate(limit_direction="both").to_numpy()

        ax.plot(np.arange(len(hours)), y_series, label=f"{y} actual")
        
    ax.set_title(f"{zone} — 2025 prediction vs past actuals (TG windows)")
    ax.set_xlabel("Hour in window (0..239)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.15)

    out = FIG_DIR / f"{zone}_2025_pred_vs_past.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("zone", help="Zone code, e.g. AECO")
    ap.add_argument("--years", nargs="*", type=int, default=[2022, 2023, 2024],
                    help="Past actual years to overlay with 2025 prediction")
    args = ap.parse_args()

    p1 = plot_2024_actual_vs_pred(args.zone)
    print(f"[plots] wrote {p1}")
    p2 = plot_2025_pred_vs_past(args.zone, years=args.years)
    print(f"[plots] wrote {p2}")


if __name__ == "__main__":
    main()