.PHONY: help predictions baseline_build rawdata_weather rawdata_weather_all weather_refresh_map weather_recent_all

# Use python3 by default
PY ?= python3 -u

help:
	@echo "make predictions                          # print one-line CSV (current model)"
	@echo "make baseline_build                        # rebuild baseline from data/raw/power/*.csv"
	@echo "make rawdata_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make rawdata_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make weather_recent_all DAYS=31            # download last N days for all zones"
	@echo "make weather_refresh_map                   # rebuild stations_map.csv"

# ---- predictions (already working) ----
predictions:
	$(PY) -m src.print_predictions

# ---- build baseline from local PJM raw files ----
baseline_build:
	$(PY) -m src.baseline_build_from_raw

# ---- weather downloads (hourly via Meteostat) ----
# one zone
rawdata_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)" ]  || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) $(ZONE)

# all zones listed in data/raw/zones_locations.csv
rawdata_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) --all

# convenience: last N days (default 31) for all zones
weather_recent_all:
	$(PY) - << 'PY'
	from datetime import date, timedelta
	from src.downloads.weather_history import main as run
	import sys
	days = int("${DAYS:-31}")
	end = date.today()
	start = end - timedelta(days=days)
	sys.argv = ["weather_history", start.isoformat(), end.isoformat(), "--all"]
	run()
	PY

# rebuild nearest-station cache (rarely needed)
weather_refresh_map:
	$(PY) - << 'PY'
	from src.downloads.weather_history import load_zones, nearest_station_point
	from pathlib import Path
	import pandas as pd
	zones = load_zones()
	rows = []
	for _, r in zones.iterrows():
		p, info = nearest_station_point(float(r['lat']), float(r['lon']))
		rows.append({"zone": r["zone"], **info})
	Path("data/raw/weather").mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows).to_csv("data/raw/weather/stations_map.csv", index=False)
	print("Refreshed data/raw/weather/stations_map.csv")
	PY
