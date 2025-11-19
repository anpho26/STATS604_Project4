"""
Refresh the most recent N days of Meteostat history for all zones.

Purpose
-------
Keep historical weather near-current without re-downloading years of data.
Typical default is N≈31 days.

Outputs
-------
- updates CSVs under `data/raw/weather/` (per zone)

CLI
---
    python -m src.downloads.weather_recent_all
    python -m src.downloads.weather_recent_all --days 45

Args
----
--days : int, optional
    Number of trailing days to refresh (default: ~31).

Notes
-----
* Writes/overwrites the recent portion per zone; older data are preserved.
* All timestamps include both UTC and **EPT**; downstream code models on EPT.
"""

from datetime import date, timedelta
import os, sys
from src.downloads.weather_history import main as run

def main():
    days = int(os.getenv("DAYS", "31"))  # default 31
    end = date.today()
    start = end - timedelta(days=days)
    sys.argv = ["weather_history", start.isoformat(), end.isoformat(), "--all"]
    run()

if __name__ == "__main__":
    main()