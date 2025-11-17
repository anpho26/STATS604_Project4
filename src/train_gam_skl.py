# src/train_gam_skl.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import json
import re
import numpy as np
import pandas as pd
from datetime import date, timedelta

from patsy import dmatrix
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Optional canonical ordering; leave empty to auto-discover from data
from src.constants import ZONES

# ------------------ paths ------------------
RAW_POWER_DIR = Path("data/raw/power")
WX_HIST_DIR   = Path("data/raw/weather")
MODEL_DIR     = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Collect per-zone CV results for a small report
_CV_ROWS: list[dict] = []

# ---------- diagnostics ----------
def _time_range(df: pd.DataFrame, col: str) -> str:
    if df.empty:
        return "—"
    t = pd.to_datetime(df[col], errors="coerce")
    return f"{t.min()} .. {t.max()} (n={len(df)})"

def diagnose_zone(zone: str, power: pd.DataFrame, temp: pd.DataFrame,
                  cutoff: pd.Timestamp, sample: int = 8) -> None:
    """
    Print step-by-step diagnostics explaining why a zone gets skipped.
    """
    import numpy as np

    print(f"[diag] === {zone} ===")
    pz = power.loc[power["zone"] == zone].copy()
    tz = temp.loc[temp["zone"] == zone].copy()

    print(f"[diag] power[{zone}]   time range: {_time_range(pz, 'datetime_beginning_ept')}")
    print(f"[diag] weather[{zone}] time range: {_time_range(tz, 'datetime_beginning_ept')}")

    # 0) immediate empties
    if pz.empty:
        print("[diag] No power rows for this zone. Missing PJM CSVs or zone label mismatch?")
    if tz.empty:
        print("[diag] No weather rows for this zone. Missing Meteostat history? Wrong zone label?")
    if pz.empty or tz.empty:
        return

    # 1) inner join (no filters)
    j0 = pz.merge(
        tz[["zone", "datetime_beginning_ept", "temp"]],
        on=["zone", "datetime_beginning_ept"], how="inner",
        suffixes=("_pw", "_wx"),
    )
    print(f"[diag] inner-join rows (exact time match): {len(j0)}")

    # If the inner join is 0, test if we might be off by timezone (~60 min)
    if len(j0) == 0:
        wx_t = tz["datetime_beginning_ept"].dropna().sort_values().to_numpy("datetime64[ns]")
        pw_t = pz["datetime_beginning_ept"].dropna().drop_duplicates().sort_values().head(50)
        if len(wx_t) and len(pw_t):
            diffs = []
            for t in pw_t:
                d = np.min(np.abs(wx_t - np.datetime64(t)))
                diffs.append(int(d.astype("timedelta64[m]") / np.timedelta64(1, "m")))
            if diffs:
                med = int(np.median(diffs))
                if med in (60, 120, 240, 300):
                    print(f"[diag] Timestamps are off by ~{med} minutes → timezone misalignment (EPT vs UTC?)")
                else:
                    print(f"[diag] Example minute offsets between power & weather (median): {med} min")

        # Show a few power times that lack a weather match
        missing = (
            pz.merge(tz[["zone", "datetime_beginning_ept"]], how="left",
                     on=["zone", "datetime_beginning_ept"], indicator=True)
              .query('_merge == "left_only"')["datetime_beginning_ept"]
              .head(sample).tolist()
        )
        if missing:
            print(f"[diag] sample power times with NO weather match: {missing}")
        # Also show a few weather times that lack a power match
        missing_w = (
            tz.merge(pz[["zone", "datetime_beginning_ept"]], how="left",
                     on=["zone", "datetime_beginning_ept"], indicator=True)
              .query('_merge == "left_only"')["datetime_beginning_ept"]
              .head(sample).tolist()
        )
        if missing_w:
            print(f"[diag] sample weather times with NO power match: {missing_w}")

    # 2) filters: fall → years → cutoff
    jf = j0[is_fall(j0["datetime_beginning_ept"])]
    jy = jf[jf["datetime_beginning_ept"].dt.year.isin([2023, 2024, 2025])]
    jc = jy[jy["datetime_beginning_ept"] <= cutoff]
    print(f"[diag] after fall filter: {len(jf)}; after year filter: {len(jy)}; after cutoff({cutoff.date()}): {len(jc)}")

    if len(j0) and len(jc) == 0:
        print("[diag] All rows removed by filters. Is your date window too tight for this zone?")
    
# ------------------ calendar helpers ------------------
def thanksgiving_date(year: int) -> date:
    """4th Thursday of November."""
    d = date(year, 11, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thu + timedelta(weeks=3)

def tg_window(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """10-day window ending Saturday of Thanksgiving week (inclusive)."""
    tg = thanksgiving_date(year)
    end = pd.Timestamp(tg + timedelta(days=2))  # Saturday
    start = end - pd.Timedelta(days=9)         # 10 consecutive days
    return start, end

def cutoff_before_window(year: int, embargo_days: int = 2) -> pd.Timestamp:
    """Last timestamp allowed for training (embargo before forecast window)."""
    start, _ = tg_window(year)
    # end of day before the embargo boundary
    return (start - pd.Timedelta(days=embargo_days)).replace(hour=23, minute=59, second=59)

def is_fall(ts: pd.Series) -> pd.Series:
    """Sep–Nov mask."""
    return ts.dt.month.isin([9, 10, 11])

def holiday_flags(ts: pd.Series) -> pd.DataFrame:
    """Veterans, Thanksgiving, Black Friday indicators."""
    idx = ts.dt.date
    years = sorted({d.year for d in idx})
    vets = {date(y, 11, 11) for y in years}
    tg   = {thanksgiving_date(y) for y in years}
    bf   = {thanksgiving_date(y) + timedelta(days=1) for y in years}
    return pd.DataFrame(
        {
            "is_veterans":      idx.isin(vets).astype(int),
            "is_thanksgiving":  idx.isin(tg).astype(int),
            "is_blackfri":      idx.isin(bf).astype(int),
        },
        index=ts.index,
    )

# ------------------ IO ------------------
def load_temp_history() -> pd.DataFrame:
    """
    Load hourly weather history with columns:
      - zone (in file or inferred from filename)
      - datetime_beginning_ept
      - temp (Celsius)
    Skips any CSV that doesn't contain temperature.
    """
    files = sorted(WX_HIST_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No weather CSVs under data/raw/weather/")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # accept either 'temp' or 'temperature_2m'
        if ("temp" not in df.columns) and ("temperature_2m" not in df.columns):
            continue  # e.g., stations_map.csv
        if "temp" not in df.columns and "temperature_2m" in df.columns:
            df = df.rename(columns={"temperature_2m": "temp"})
        if "datetime_beginning_ept" not in df.columns:
            raise KeyError(f"{f}: missing 'datetime_beginning_ept'")
        if "zone" not in df.columns:
            z = _infer_zone_from_name(f)
            if z is None:
                continue
            df["zone"] = z
        df["datetime_beginning_ept"] = pd.to_datetime(
        df["datetime_beginning_ept"], format="mixed", errors="coerce"
)
        dfs.append(df[["zone", "datetime_beginning_ept", "temp"]])
    if not dfs:
        raise FileNotFoundError("No usable weather CSVs with temperature in data/raw/weather/")
    out = pd.concat(dfs, ignore_index=True).dropna()
    out["zone"] = out["zone"].astype(str)
    return out

def attach_temp_day_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns ['zone','datetime_beginning_ept','temp'].
    Adds one column 'temp_day_mean' (same value for all 24 hours of a date).
    """
    out = df.copy()
    out["__date"] = out["datetime_beginning_ept"].dt.date
    out["temp_day_mean"] = out.groupby(["zone","__date"])["temp"].transform("mean")
    return out.drop(columns="__date")

_ZONE_RE = re.compile(r"(?:meteostat_|openmeteo_(?:forecast|history)_)?([A-Z0-9]+)")

def _infer_zone_from_name(path: Path) -> str | None:
    m = _ZONE_RE.search(path.stem)
    return m.group(1) if m else None

def load_power() -> pd.DataFrame:
    """Load PJM metered load using ONLY 'load_area' (preferred) as the area ID."""
    files = sorted(RAW_POWER_DIR.glob("hrl_load_metered_*.csv"))
    if not files:
        raise FileNotFoundError("No PJM CSVs under data/raw/power/")

    parts = []
    for f in files:
        df = pd.read_csv(f)

        # --- choose the identifier column (prefer load_area) ---
        if "load_area" in df.columns:
            zone_series = df["load_area"].astype(str).str.strip()
        elif "load area" in df.columns:
            zone_series = df["load area"].astype(str).str.strip()
        elif "zone" in df.columns:         # fallback if no load_area
            zone_series = df["zone"].astype(str).str.strip()
        elif "zones" in df.columns:
            zone_series = df["zones"].astype(str).str.strip()
        else:
            raise ValueError(f"{f}: missing 'load_area' (or 'zone') column")

        # --- MW column (allow 'load' as alias) ---
        if "mw" not in df.columns and "load" in df.columns:
            df = df.rename(columns={"load": "mw"})
        if "mw" not in df.columns:
            raise ValueError(f"{f}: missing 'mw' (or 'load') column")

        # --- timestamp column must be 'datetime_beginning_ept' ---
        if "datetime_beginning_ept" not in df.columns:
            raise KeyError(f"{f}: missing 'datetime_beginning_ept'")
        ts = pd.to_datetime(df["datetime_beginning_ept"], format="mixed", errors="coerce")

        # build a clean 3-column frame to avoid duplicate names
        parts.append(pd.DataFrame({
            "zone": zone_series,
            "datetime_beginning_ept": ts,
            "mw": df["mw"].astype(float),
        }))

    out = pd.concat(parts, ignore_index=True).dropna()
    return out

# ------------------ features ------------------
def fourier_cyclic(x: pd.Series, period: float, K: int, prefix: str) -> pd.DataFrame:
    """sin/cos Fourier basis for cyclic effects."""
    x = x.to_numpy(dtype=float)
    cols = {}
    for k in range(1, K + 1):
        cols[f"{prefix}_s{k}"] = np.sin(2 * np.pi * k * x / period)
        cols[f"{prefix}_c{k}"] = np.cos(2 * np.pi * k * x / period)
    return pd.DataFrame(cols)

def temp_spline(temp: pd.Series, df: int = 6, prefix: str = "temp") -> pd.DataFrame:
    """Cubic B-spline basis for temperature via patsy."""
    B = dmatrix(
        f"bs(x, df={df}, degree=3, include_intercept=False)",
        {"x": temp.to_numpy(dtype=float)},
        return_type="dataframe",
    )
    B.columns = [f"{prefix}_b{i+1}" for i in range(B.shape[1])]
    return B

def build_design(df: pd.DataFrame, K_hour: int = 3, K_doy: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Design matrix with:
      - hour Fourier (K_hour pairs)
      - day-of-year Fourier (K_doy pairs)
      - weekend/holiday dummies
      - year dummies
      - ONE numeric feature: temp_day_mean  (daily mean temp, broadcast to hours)
    """
    t = pd.to_datetime(df["datetime_beginning_ept"])
    hour = t.dt.hour
    doy  = t.dt.dayofyear
    dow  = t.dt.dayofweek  # Mon=0

    X_parts: List[pd.DataFrame] = []

    # cyclic seasonality
    X_parts.append(fourier_cyclic(hour, period=24,     K=K_hour, prefix="hr"))
    X_parts.append(fourier_cyclic(doy,  period=365.25, K=K_doy,  prefix="doy"))

    # >>> daily mean temperature (single linear feature)
    if "temp_day_mean" not in df.columns:
        raise KeyError("build_design expects column 'temp_day_mean'")
    X_parts.append(pd.DataFrame({"temp_day_mean": df["temp_day_mean"].to_numpy(dtype=float)}))

    # weekday/weekend + holidays
    dums = pd.DataFrame({
        "dow_1": (dow==1).astype(int),
        "dow_2": (dow==2).astype(int),
        "dow_3": (dow==3).astype(int),
        "dow_4": (dow==4).astype(int),
        "dow_5": (dow==5).astype(int),
        "dow_6": (dow==6).astype(int),
        "is_weekend": dow.isin([5,6]).astype(int),
    }, index=df.index)
    hol = holiday_flags(t)
    X_parts += [dums.reset_index(drop=True), hol.reset_index(drop=True)]

    # year dummies (2024 baseline)
    yy = t.dt.year
    years = pd.DataFrame({
        "yr_2022": (yy==2022).astype(int),
        "yr_2023": (yy==2023).astype(int),
        "yr_2025": (yy==2025).astype(int),
    }, index=df.index)
    X_parts.append(years.reset_index(drop=True))

    X = pd.concat(X_parts, axis=1)
    return X, list(X.columns)

# ------------------ offsets & bias ------------------
def per_row_hour_offset(df: pd.DataFrame) -> pd.Series:
    """log1p rolling mean of previous 14 same-hour loads (zone-specific df)."""
    g = df.sort_values("datetime_beginning_ept").copy()
    g["hour"] = g["datetime_beginning_ept"].dt.hour
    out = pd.Series(index=g.index, dtype=float)
    for h in range(24):
        m = g["hour"] == h
        out.loc[m] = g.loc[m, "mw"].shift(1).rolling(14, min_periods=7).mean()
    return np.log1p(out).reindex(df.index)

def fixed_hour_offset(past: pd.DataFrame) -> Dict[int, float]:
    """Fixed hour means (log1p) from the most recent 14 same-hour observations."""
    tmp = past.sort_values("datetime_beginning_ept").copy()
    tmp["hour"] = tmp["datetime_beginning_ept"].dt.hour
    means: Dict[int, float] = {}
    for h in range(24):
        s = tmp.loc[tmp["hour"] == h, "mw"].tail(14)
        means[h] = float(np.log1p(s.mean())) if len(s) else float("nan")
    return means

def attach_fixed_offset(df_future: pd.DataFrame, hour_means: Dict[int, float]) -> pd.Series:
    hrs = df_future["datetime_beginning_ept"].dt.hour
    vals = hrs.map(hour_means).astype(float)
    if np.isnan(vals).any():
        fill = float(np.nanmean(list(hour_means.values())))
        vals = vals.fillna(fill if not np.isnan(fill) else 0.0)
    return pd.Series(vals.values, index=df_future.index, dtype=float)

def mean_log_residual_last14(train: pd.DataFrame, yhat_log: pd.Series) -> float:
    """Average (log1p(y) − yhat_log) on the last ~14 days to set a bias delta."""
    last_cut = train["datetime_beginning_ept"].max() - pd.Timedelta(days=14)
    m = train["datetime_beginning_ept"] > last_cut
    return float((np.log1p(train.loc[m, "mw"]) - yhat_log.loc[m]).mean())

# ------------------ training & selection ------------------
@dataclass
class BestConfig:
    alpha: float

def _daily_peak_hour(s: pd.Series) -> int:
    # s indexed by hour already; return argmax hour (0-23)
    return int(s.idxmax())

def peak_hour_accuracy_pm1(df: pd.DataFrame) -> float:
    """
    df columns: datetime_beginning_ept, mw, mw_pred
    Return fraction of days where |pred_peak_hour - true_peak_hour| <= 1.
    """
    d = df.copy()
    d["date"] = d["datetime_beginning_ept"].dt.date
    d["hour"] = d["datetime_beginning_ept"].dt.hour

    # pick the row index of the daily max (actual & predicted)
    idx_act = d.groupby("date")["mw"].idxmax()
    idx_pre = d.groupby("date")["mw_pred"].idxmax()

    act = d.loc[idx_act, ["date", "hour"]].rename(columns={"hour": "h_act"}).set_index("date")
    pre = d.loc[idx_pre, ["date", "hour"]].rename(columns={"hour": "h_pre"}).set_index("date")

    both = act.join(pre, how="inner")
    if len(both) == 0:
        return 0.0
    return float((both["h_act"] - both["h_pre"]).abs().le(1).mean())


def peak_day_cost(df: pd.DataFrame) -> int:
    """
    Cost for choosing exactly 2 peak days over the window.
    FN (missed actual peak) = 4, FP (extra predicted peak) = 1.
    """
    d = df.copy()
    d["date"] = d["datetime_beginning_ept"].dt.date

    daymax_act = d.groupby("date")["mw"].max()
    daymax_pre = d.groupby("date")["mw_pred"].max()

    A = set(daymax_act.nlargest(2).index)   # actual peak days
    P = set(daymax_pre.nlargest(2).index)   # predicted peak days

    fn = len(A - P)   # missed actual peaks
    fp = len(P - A)   # extra predicted peaks
    return 4 * fn + 1 * fp

def pick_alpha_for_zone(zone: str, power: pd.DataFrame, temp: pd.DataFrame) -> BestConfig:
    start24, end24 = tg_window(2024)
    cutoff24 = cutoff_before_window(2024, embargo_days=2)


    merged = power.merge(temp, on=["zone","datetime_beginning_ept"], how="inner")

    train = merged[
        (merged["zone"] == zone)
        & is_fall(merged["datetime_beginning_ept"])
        & (merged["datetime_beginning_ept"].dt.year.isin([2022, 2023, 2024]))
        & (merged["datetime_beginning_ept"] <= cutoff24)
    ].sort_values("datetime_beginning_ept")
    train = attach_temp_day_mean(train)


    # make the window [start24, end24_excl)
    start24 = pd.Timestamp(start24).normalize()
    end24_excl = pd.Timestamp(end24).normalize() + pd.Timedelta(days=1)

    valid = (merged[
        (merged["zone"] == zone)
        & (merged["datetime_beginning_ept"] >= start24)
        & (merged["datetime_beginning_ept"] <  end24_excl)   # <-- strictly less than next midnight
    ].sort_values("datetime_beginning_ept"))
    valid = attach_temp_day_mean(valid)

    if len(train) < 1500 or len(valid) < 100:
        print(f"[GAM-skl][cv] {zone}: alpha*=10.0 (fallback), rmse_valid=NA, train_rows={len(train)}, valid_rows={len(valid)}")
        return BestConfig(alpha=10.0)

    # offsets and design (train)
    off_tr = per_row_hour_offset(train)
    y_tr = np.log1p(train["mw"].values) - off_tr.values
    X_tr, _ = build_design(train)

    # mask NA/inf
    mask_tr = (
        np.isfinite(y_tr) &
        np.isfinite(off_tr.values) &
        np.isfinite(X_tr.to_numpy()).all(axis=1)
    )
    train, off_tr, y_tr, X_tr = train.iloc[mask_tr], off_tr.iloc[mask_tr], y_tr[mask_tr], X_tr.iloc[mask_tr]

    # validation design + fixed-hour offset
    hour_means = fixed_hour_offset(train)
    off_va = attach_fixed_offset(valid, hour_means)
    X_va, _ = build_design(valid)

    alphas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_alpha, best_rmse = None, float("inf")
    best_pred_va = None

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_tr.values)

    for a in alphas:
        model = Ridge(alpha=a, fit_intercept=True)
        model.fit(Xs, y_tr)

        # bias tweak from last 14 train days
        yhat_tr_log = model.predict(Xs) + off_tr.values
        delta = mean_log_residual_last14(train, pd.Series(yhat_tr_log, index=train.index))

        yhat_va_log = model.predict(scaler.transform(X_va.values)) + off_va.values + delta
        yhat_va = np.expm1(yhat_va_log)

        rmse = float(np.sqrt(np.mean((valid["mw"].values - yhat_va) ** 2)))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_alpha  = a
            best_pred_va = yhat_va  # keep the predictions for metrics

    # --- compute the two requested metrics on the best alpha ---
    df_va = valid.copy()
    df_va["mw_pred"] = best_pred_va
    acc_peakhr_pm1 = peak_hour_accuracy_pm1(df_va)      # proportion in [0,1]
    loss_peakday   = peak_day_cost(df_va)               # integer cost

    print(f"[GAM-skl][cv] {zone}: alpha*={best_alpha}, rmse_valid={best_rmse:.1f}, "
          f"acc_peakhr±1={acc_peakhr_pm1:.3f}, loss_peakday={loss_peakday}, "
          f"train_rows={len(train)}, valid_rows={len(valid)}")

    # If you already assemble a CV CSV, append a row like this where you build it:
    global _CV_ROWS  # define this list at module top if you like
    try:
        _CV_ROWS.append({
            "zone": zone,
            "alpha": best_alpha,
            "rmse_valid": best_rmse,
            "acc_peakhour_pm1": acc_peakhr_pm1,
            "loss_peakday": loss_peakday,
            "train_rows": len(train),
            "valid_rows": len(valid),
            "train_start": train["datetime_beginning_ept"].min(),
            "train_end":   train["datetime_beginning_ept"].max(),
            "valid_start": valid["datetime_beginning_ept"].min(),
            "valid_end":   valid["datetime_beginning_ept"].max(),
        })
    except NameError:
        pass

    return BestConfig(alpha=best_alpha if best_alpha is not None else 10.0)

def train_zone_2025(zone: str, cfg: BestConfig, power: pd.DataFrame, temp: pd.DataFrame) -> None:
    """
    Final fit to produce 2025 models:
      * Train: Sep–Nov 2023–2025 up to the 2025 cutoff
    """
    cutoff25 = cutoff_before_window(2025, embargo_days=2)

    merged = power.merge(temp, on=["zone","datetime_beginning_ept"], how="inner")
    df = merged[
        (merged["zone"] == zone)
        & is_fall(merged["datetime_beginning_ept"])
        & (merged["datetime_beginning_ept"].dt.year.isin([2023, 2024, 2025]))
        & (merged["datetime_beginning_ept"] <= cutoff25)
    ].sort_values("datetime_beginning_ept")
    df = attach_temp_day_mean(df)

    if len(df) < 1200:
        print(f"[GAM-skl] skip {zone} (rows={len(df)})")
        # print detailed diagnostics so we can fix quickly
        diagnose_zone(zone, power, temp, cutoff25)
        return

    off = per_row_hour_offset(df)
    y = np.log1p(df["mw"].values) - off.values
    X, cols = build_design(df)

    # Drop rows with NaNs/Infs before fitting the pipeline
    X_np   = X.to_numpy()
    mask   = (
        np.isfinite(y)
        & np.isfinite(off.values)
        & np.isfinite(X_np).all(axis=1)
    )

    df   = df.iloc[mask]
    off  = off.iloc[mask]
    y    = y[mask]
    X    = X.iloc[mask]

    scaler = StandardScaler(with_mean=True, with_std=True)
    model = Ridge(alpha=cfg.alpha, fit_intercept=True)
    pipe = Pipeline([("scaler", scaler), ("ridge", model)])
    pipe.fit(X.values, y)

    # ----- compute training metrics -----
    yhat_log_tr = pipe.predict(X.values) + off.values          # log-space preds + offset
    yhat_tr     = np.expm1(yhat_log_tr)                        # back to MW
    train_rmse  = float(np.sqrt(np.mean((df["mw"].values - yhat_tr) ** 2)))
    delta       = mean_log_residual_last14(df, pd.Series(yhat_log_tr, index=df.index))

    # ----- metadata for forecasting -----
    meta = {
        "columns": cols,
        "alpha": cfg.alpha,
        "cutoff": cutoff25.isoformat(),
        "hour_means_log1p": fixed_hour_offset(df),
        "delta": delta,
    }

    # save
    dump(pipe, MODEL_DIR / f"gam_skl_{zone}.joblib")
    (MODEL_DIR / f"gam_skl_{zone}.meta.json").write_text(json.dumps(meta))

    print(f"[GAM-skl][fit] {zone}: rows={len(df)}, train_rmse={train_rmse:.1f}, delta={delta:.3f}")

# ------------------ entry ------------------
def main() -> None:
    power = load_power()
    temp  = load_temp_history()

    zones = ZONES if ZONES else sorted(power["zone"].unique())

    print("[GAM-skl] selecting alpha via 2024 backtest …")
    cfgs: Dict[str, BestConfig] = {}
    for z in zones:
        cfgs[z] = pick_alpha_for_zone(z, power, temp)
        print(f"[GAM-skl] {z}: alpha={cfgs[z].alpha}")

    # Write a small CV summary CSV
    if _CV_ROWS:
        rep = pd.DataFrame(_CV_ROWS)
        out_csv = MODEL_DIR / "gam_skl_cv_report.csv"
        rep.to_csv(out_csv, index=False)
        print(f"[GAM-skl] wrote CV summary -> {out_csv}")

    print("[GAM-skl] training final 2025 models …")
    for z in zones:
        train_zone_2025(z, cfgs[z], power, temp)

    manifest = {z: {"alpha": cfgs[z].alpha} for z in zones}
    (MODEL_DIR / "gam_skl_manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()