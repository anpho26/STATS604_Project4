SHELL := /bin/bash
.ONESHELL:

.PHONY: help predictions baseline_build \
        rawdata_weather rawdata_weather_all weather_recent_all weather_refresh_map \
        forecast_weather forecast_weather_all forecast_next_all \
		train_lgbm forecast_10d forecast_next_all daily_10day \

# use python3 by default (respects your venv)
PY ?= python3 -u

help:
	@echo "make predictions                               # print one-line CSV using current model"
	@echo "make baseline_build                             # rebuild baseline from data/raw/power/*.csv"
	@echo "make rawdata_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make rawdata_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make weather_recent_all DAYS=31                 # history: last N days for ALL zones"
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
	$(PY) -m src.downloads.weather_recent_all
	
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


# Just fetch Open-Meteo forecast for tomorrow..+9 (ALL zones)
forecast_next_all:
	$(PY) -m src.downloads.forecast_next_all

# Fetch forecast AND run the 10-day load predictions
daily_10day:
	$(PY) -m src.pipelines.daily_10day

train_lgbm:
	$(PY) -m src.train_lgbm

# Example: make forecast_10d START=2025-11-20 END=2025-11-29
forecast_10d:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.forecast_10day $(START) $(END)