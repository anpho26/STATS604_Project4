SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := all

.PHONY: all clean rawdata help predictions baseline_build \
        rawdata_weather rawdata_weather_all weather_recent_all weather_refresh_map \
        forecast_weather forecast_weather_all forecast_next_all \
        train_lgbm forecast_10d daily_10day

# use python3 by default (respects your venv)
PY ?= python3 -u

# ---- DEFAULT: make ----
# Run all analyses that build artifacts, but DO NOT download raw data or emit today's predictions
all: baseline_build train_lgbm
	@echo "[all] baseline + models are up to date."

help:
	@echo "make                     # runs all analyses (no downloads, no current predictions)"
	@echo "make clean               # delete outputs/derived files; keep code & raw data"
	@echo "make rawdata             # delete & re-download raw data (weather), keep power raw"
	@echo "make predictions         # print required one-line CSV for 'tomorrow'"
	@echo "make baseline_build      # rebuild baseline from data/raw/power/*.csv"
	@echo "make rawdata_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make rawdata_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make weather_recent_all  # history: last N days for ALL zones"
	@echo "make forecast_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "make forecast_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "make daily_10day         # fetch forecast (tomorrow..+9) and run 10-day predictions"

# ----- predictions -----
predictions:
	@$(PY) -m src.print_predictions

# ----- CLEAN -----
# Keep code and raw data; remove models, predictions, processed/interim, caches, notebooks' checkpoints
CLEAN_DIRS := data/models data/predictions_10day data/processed data/interim \
              .pytest_cache **/__pycache__
clean:
	@echo "[clean] removing derived artifacts…"
	rm -rf $(CLEAN_DIRS)
	# keep raw data:
	#   data/raw/power (your PJM CSVs)
	#   data/raw/weather (meteostat history)
	#   data/raw/weather_forecast (optional to keep)
	@echo "[clean] done."

# ----- RAW DATA (assignment-style) -----
# Delete and re-download raw data (we treat 'raw' as Meteostat weather; PJM power is already local).
RAW_WX_DIR := data/raw/weather
RAW_WXF_DIR := data/raw/weather_forecast

rawdata:
	@echo "[rawdata] removing weather raw directories…"
	rm -rf $(RAW_WX_DIR) $(RAW_WXF_DIR)
	mkdir -p $(RAW_WX_DIR)
	@echo "[rawdata] re-downloading Meteostat history for ALL zones since 2021…"
	$(PY) -m src.downloads.weather_history 2021-01-01 $$($(PY) -c 'from datetime import date; print(date.today().isoformat())') --all
	@echo "[rawdata] done."

# ----- WEATHER HISTORY (Meteostat) -----
rawdata_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)" ]  || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) $(ZONE)

rawdata_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) --all

weather_recent_all:
	$(PY) -m src.downloads.weather_recent_all

# ----- WEATHER FORECAST (Open-Meteo) -----
forecast_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)" ]  || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) $(ZONE)

forecast_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) --all

forecast_next_all:
	$(PY) -m src.downloads.forecast_next_all

# ----- MODELS & 10-day pipeline -----
train_lgbm:
	$(PY) -m src.train_lgbm

forecast_10d:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)" ]   || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.forecast_10day $(START) $(END)

daily_10day:
	$(PY) -m src.pipelines.daily_10day