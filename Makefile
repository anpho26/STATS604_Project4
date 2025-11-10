.PHONY: help predictions baseline_build \
        rawdata_weather rawdata_weather_all weather_recent_all weather_all_since_2021 weather_refresh_map \
        forecast_weather forecast_weather_all forecast_next_all

# use python3 by default (respects your venv)
PY ?= python3 -u

help:
	@echo "make predictions                               # print one-line CSV using current model"
	@echo "make baseline_build                             # rebuild baseline from data/raw/power/*.csv"
	@echo "make rawdata_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make rawdata_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make weather_recent_all DAYS=31                 # history: last N days for ALL zones"
	@echo "make weather_all_since_2021                     # history: ALL zones since 2021 (month chunks)"
	@echo "make forecast_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make forecast_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make forecast_next_all DAYS=7                   # forecast: next N days for ALL zones"

# ----- predictions -----
predictions:
	$(PY) -m src.print_predictions

# ----- build baseline from local PJM raw files -----
baseline_build:
	$(PY) -m src.baseline_build_from_raw

# ----- WEATHER HISTORY (Meteostat) -----
# one zone
rawdata_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)" ]  || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) $(ZONE)

# all zones
rawdata_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) --all

# last N days (default 31) for ALL zones
weather_recent_all:
	$(PY) - << 'PY'
	from datetime import date, timedelta
	import sys
	from src.downloads.weather_history import main as run
	days = int("${DAYS:-31}")
	end = date.today()
	start = end - timedelta(days=days)
	sys.argv = ["weather_history", start.isoformat(), end.isoformat(), "--all"]
	run()
	PY

# all zones since 2021 in month chunks (resume-safe & smaller requests)
weather_all_since_2021:
	$(PY) - << 'PY'
	from datetime import date
	from dateutil.relativedelta import relativedelta
	import sys
	from src.downloads.weather_history import main as run
	start = date(2021, 1, 1)
	today = date.today()
	cur = start
	while cur < today:
		nxt = min(cur + relativedelta(months=1), today)
		print(f"[batch] {cur}..{nxt}")
		sys.argv = ["weather_history", cur.isoformat(), nxt.isoformat(), "--all"]
		run()
		cur = nxt
	PY

# ----- WEATHER FORECAST (Open-Meteo) -----
# one zone
forecast_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)" ]  || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) $(ZONE)

# all zones
forecast_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) --all

# next N days (default 7) for ALL zones
forecast_next_all:
	$(PY) - << 'PY'
	from datetime import date, timedelta
	import sys
	from src.downloads.weather_forecast import main as run
	days = int("${DAYS:-7}")
	start = date.today()
	end = start + timedelta(days=days)
	sys.argv = ["weather_forecast", start.isoformat(), end.isoformat(), "--all"]
	run()
	PY
	MAKE