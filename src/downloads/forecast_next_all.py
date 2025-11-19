"""
Convenience: fetch the next 10 days of Open-Meteo forecasts for all zones.

Purpose
-------
Compute a fixed window of [tomorrow .. tomorrow+9 days] and call the
Open-Meteo fetcher for all zones. Intended for the daily prediction pipeline.

Outputs
-------
- files under `data/raw/weather_forecast/` for each zone

CLI
---
    python -m src.downloads.forecast_next_all

Notes
-----
* Horizon length defaults to 10 days. You can change it by editing the module
  or via an environment variable if implemented (e.g., `DAYS=10`).
* This script is idempotent; re-running updates the same date window.
"""


from datetime import date, timedelta
import sys
from src.downloads.weather_forecast import main as wx_main

def main():
    start = date.today() + timedelta(days=1)
    end   = start + timedelta(days=9)
    sys.argv = ["weather_forecast", start.isoformat(), end.isoformat(), "--all"]
    wx_main()

if __name__ == "__main__":
    main()