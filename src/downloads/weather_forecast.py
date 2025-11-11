# Fetch hourly forecast from Open-Meteo for zones in data/raw/zones_locations.csv
# Output: data/raw/weather_forecast/openmeteo_forecast_<ZONE>_<START>_<END>.csv
# Columns start with PJM-style time headers:
#   datetime_beginning_utc, datetime_beginning_ept

from pathlib import Path
from datetime import datetime
import sys
import requests
import pandas as pd

ZONES_CSV = Path("data/raw/zones_locations.csv")   # columns: zone, lat, lon
OUT_DIR   = Path("data/raw/weather_forecast")
LOCAL_TZ  = "US/Eastern"                           # PJM's EPT

# default hourly variables (edit as you like)
DEFAULT_VARS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "weathercode",
    "windspeed_10m", "windgusts_10m", "winddirection_10m",
    "pressure_msl", "cloudcover"
]

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def load_zones() -> pd.DataFrame:
    df = pd.read_csv(ZONES_CSV)
    return df[["zone", "lat", "lon"]].copy()

def fetch_openmeteo_hourly(lat: float, lon: float, start: datetime, end: datetime,
                           hourly_vars=None) -> pd.DataFrame:
    """Call Open-Meteo forecast API with UTC timestamps; return DataFrame indexed by UTC."""
    if hourly_vars is None:
        hourly_vars = DEFAULT_VARS

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",                    # return times in UTC
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hz = data.get("hourly", {})
    times = hz.get("time", [])
    if not times:
        return pd.DataFrame()  # empty

    d = {"time_utc": pd.to_datetime(times, utc=True)}
    for v in hourly_vars:
        d[v] = hz.get(v, [None] * len(times))
    df = pd.DataFrame(d).set_index("time_utc")
    
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

def save_with_pjm_times(df: pd.DataFrame, dest: Path):
    """Add PJM-style timestamp columns and save CSV."""
    if df.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["datetime_beginning_utc", "datetime_beginning_ept"]).to_csv(dest, index=False)
        return

    # Build UTC & EPT strings (naive, PJM-like formatting)
    idx_utc = df.index.tz_convert("UTC").tz_localize(None)
    idx_ept = df.index.tz_convert(LOCAL_TZ).tz_localize(None)

    fmt_try = "%-m/%-d/%Y %-I:%M:%S %p"   # mac/linux
    fmt_win = "%#m/%#d/%Y %#I:%M:%S %p"   # windows
    try:
        utc_str = pd.Series(idx_utc).dt.strftime(fmt_try)
        ept_str = pd.Series(idx_ept).dt.strftime(fmt_try)
    except ValueError:
        utc_str = pd.Series(idx_utc).dt.strftime(fmt_win)
        ept_str = pd.Series(idx_ept).dt.strftime(fmt_win)

    out = df.reset_index(drop=True)
    out.insert(0, "datetime_beginning_utc", utc_str.values)
    out.insert(1, "datetime_beginning_ept", ept_str.values)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.downloads.weather_forecast START END [--all | ZONE ...]")
        sys.exit(2)

    start, end = parse_date(sys.argv[1]), parse_date(sys.argv[2])
    targets = sys.argv[3:]

    zones = load_zones()
    zones_sel = zones if (not targets or targets == ["--all"]) else zones[zones["zone"].isin(targets)]

    for _, r in zones_sel.iterrows():
        df = fetch_openmeteo_hourly(r["lat"], r["lon"], start, end)
        dest = OUT_DIR / f"openmeteo_forecast_{r['zone']}_{start.date()}_{end.date()}.csv"
        save_with_pjm_times(df, dest)
        print(f"[forecast] wrote {dest}")

if __name__ == "__main__":
    main()