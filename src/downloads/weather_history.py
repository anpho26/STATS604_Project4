"""
Meteostat hourly weather history (EPT).

Purpose
-------
Fetch hourly weather history from Meteostat for one zone or all zones across a
date range. The script writes per-zone CSVs with both UTC and EPT timestamps
(we model on `datetime_beginning_ept` downstream).

Inputs
------
- `data/raw/zones_locations.csv`  (zone code → lat/lon)

Outputs
-------
- `data/raw/weather/meteostat_{ZONE}_{START}_{END}.csv`

Columns (typical)
-----------------
`datetime_beginning_utc, datetime_beginning_ept, temp, dwpt, rhum, prcp, snow,
 wdir, wspd, wpgt, pres, tsun, coco`

CLI
---
From repo root:

    # One zone
    python -m src.downloads.weather_history 2021-01-01 2025-11-10 AECO

    # All zones listed in zones_locations.csv
    python -m src.downloads.weather_history 2021-01-01 2025-11-10 --all

Args
----
start : YYYY-MM-DD
end   : YYYY-MM-DD (inclusive)
zone  : str, optional
--all : flag, fetch for all zones instead of a single zone

Notes
-----
* All files include both UTC and **EPT** (Eastern Prevailing Time).
* Rate limiting: if you hit HTTP 429, re-run after a short pause.
* Re-running will append/overwrite the same output file name for the range.
"""

from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
from meteostat import Stations, Hourly, Point

ZONES_CSV = Path("data/raw/zones_locations.csv")   # columns: zone, lat, lon
OUT_DIR   = Path("data/raw/weather")
MAP_CSV   = OUT_DIR / "stations_map.csv"
LOCAL_TZ  = "US/Eastern"

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def load_zones() -> pd.DataFrame:
    df = pd.read_csv(ZONES_CSV)
    return df[["zone", "lat", "lon"]].copy()

def nearest_station_point(lat: float, lon: float):
    """Return a Point at the nearest station coordinates (+ info dict)."""
    st = Stations().nearby(lat, lon).fetch(1)
    if st.empty:
        return Point(lat, lon), {"source": "fallback", "station_lat": lat, "station_lon": lon}
    row = st.reset_index().iloc[0]
    s_lat = float(row["latitude"]); s_lon = float(row["longitude"])
    info = {
        "source": "nearest",
        "station_lat": s_lat, "station_lon": s_lon,
        "station_id": str(row["id"]) if "id" in row else None,
        "name": row.get("name", None),
        "distance_km": float(row.get("distance", float("nan"))),
    }
    return Point(s_lat, s_lon), info

def fetch_hourly(zone: str, point: Point, start: datetime, end: datetime) -> Path:
    """
    Save a CSV which begins with:
      - datetime_beginning_utc  (e.g., 1/1/2016 5:00:00 AM)
      - datetime_beginning_ept  (e.g., 1/1/2016 12:00:00 AM)
    followed by all Meteostat hourly columns.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"meteostat_{zone}_{start.date()}_{end.date()}.csv"

    # Meteostat Hourly is UTC by default
    df = Hourly(point, start, end).fetch()

    # If empty, still write headers for consistency
    if df.empty:
        pd.DataFrame(columns=["datetime_beginning_utc", "datetime_beginning_ept"]).to_csv(dest, index=False)
        return dest

    # Ensure tz-aware (UTC), then make both UTC & Eastern (EPT) views
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")

    idx_utc = df.index.tz_convert("UTC").tz_localize(None)
    idx_ept = df.index.tz_convert("US/Eastern").tz_localize(None)

    # Cross-platform timestamp formatting like your PJM files
    # (%-m etc. works on mac/linux; %#m works on Windows)
    fmt_try = "%-m/%-d/%Y %-I:%M:%S %p"
    fmt_win = "%#m/%#d/%Y %#I:%M:%S %p"
    try:
        utc_str = pd.Series(idx_utc).dt.strftime(fmt_try)
        ept_str = pd.Series(idx_ept).dt.strftime(fmt_try)
    except ValueError:
        utc_str = pd.Series(idx_utc).dt.strftime(fmt_win)
        ept_str = pd.Series(idx_ept).dt.strftime(fmt_win)

    out = df.reset_index(drop=True)
    out.insert(0, "datetime_beginning_utc", utc_str.values)
    out.insert(1, "datetime_beginning_ept", ept_str.values)

    out.to_csv(dest, index=False)
    return dest

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.downloads.weather_history START END [--all | ZONE ...]")
        sys.exit(2)

    start, end = parse_date(sys.argv[1]), parse_date(sys.argv[2])
    targets = sys.argv[3:]

    zones = load_zones()
    zones_sel = zones if (not targets or targets == ["--all"]) else zones[zones["zone"].isin(targets)]

    rows = []
    for _, r in zones_sel.iterrows():
        p, info = nearest_station_point(float(r["lat"]), float(r["lon"]))
        rows.append({"zone": r["zone"], **info})
        path = fetch_hourly(r["zone"], p, start, end)
        print(f"[weather] wrote {path}")

    pd.DataFrame(rows).to_csv(MAP_CSV, index=False)
    print(f"[weather] stations map -> {MAP_CSV}")

if __name__ == "__main__":
    main()