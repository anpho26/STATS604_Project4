SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
.DEFAULT_GOAL := all

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PY ?= python3 -u

RAW_WX_DIR  := data/raw/weather
RAW_WXF_DIR := data/raw/weather_forecast
MODEL_DIR   := data/models
PRED_DIR    := data/predictions_10day

# Default 10-day window: tomorrow .. tomorrow+9  (inclusive by date)
START ?= $(shell $(PY) -c 'from datetime import date,timedelta; s=date.today()+timedelta(days=1); print(s.isoformat())')
END   ?= $(shell $(PY) -c 'from datetime import date,timedelta; s=date.today()+timedelta(days=1); print((s+timedelta(days=9)).isoformat())')

# ------------------------------------------------------------
# Phony targets
# ------------------------------------------------------------
.PHONY: all help clean \
        rawdata rawdata_weather rawdata_weather_all weather_recent_all \
        forecast_weather forecast_weather_all forecast_next_all \
        train_gam_skl forecast_10d_gam predictions daily_10day vars

# ------------------------------------------------------------
# Top-level
# ------------------------------------------------------------
all: train_gam_skl
	@echo "[all] GAM models up to date in $(MODEL_DIR)"

help:
	@echo "Usage:"
	@echo "  make                         # train GAM models (no downloads)"
	@echo "  make clean                   # remove models/predictions/processed"
	@echo "  make rawdata                 # re-download weather history since 2021"
	@echo "  make rawdata_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "  make rawdata_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "  make weather_recent_all      # last N days history for ALL zones"
	@echo "  make forecast_weather START=YYYY-MM-DD END=YYYY-MM-DD ZONE=AECO"
	@echo "  make forecast_weather_all START=YYYY-MM-DD END=YYYY-MM-DD"
	@echo "  make forecast_next_all       # convenience pull for next 10 days"
	@echo "  make train_gam_skl           # fit sklearn GAM-like ridge models"
	@echo "  make forecast_10d_gam        # write 10-day GAM predictions (uses START/END)"
	@echo "  make predictions             # print one-line CSV for tomorrow"
	@echo "  make daily_10day             # fetch forecasts for (START..END) then predict"
	@echo "  make vars                    # show the resolved START/END defaults"

vars:
	@echo "START=$(START)"
	@echo "END=$(END)"

# ------------------------------------------------------------
# Housekeeping
# ------------------------------------------------------------
CLEAN_DIRS := $(MODEL_DIR) $(PRED_DIR) data/processed data/interim \
              .pytest_cache **/__pycache__
clean:
	@echo "[clean] removing: $(CLEAN_DIRS)"
	rm -rf $(CLEAN_DIRS)
	@echo "[clean] raw data kept: data/raw/power, $(RAW_WX_DIR), $(RAW_WXF_DIR)"

# ------------------------------------------------------------
# Weather HISTORY (Meteostat)
# ------------------------------------------------------------
rawdata:
	@echo "[rawdata] nuking $(RAW_WX_DIR) and $(RAW_WXF_DIR)"
	rm -rf $(RAW_WX_DIR) $(RAW_WXF_DIR)
	mkdir -p $(RAW_WX_DIR)
	@echo "[rawdata] downloading Meteostat since 2021-01-01 for ALL zones…"
	$(PY) -m src.downloads.weather_history 2021-01-01 $$($(PY) -c 'from datetime import date; print(date.today().isoformat())') --all

rawdata_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)"   ] || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)"  ] || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) $(ZONE)

rawdata_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)"   ] || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_history $(START) $(END) --all

weather_recent_all:
	$(PY) -m src.downloads.weather_recent_all

# ------------------------------------------------------------
# Weather FORECAST (Open-Meteo)
# ------------------------------------------------------------
forecast_weather:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)"   ] || (echo "END=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(ZONE)"  ] || (echo "ZONE=<code> required (e.g., AECO)" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) $(ZONE)

forecast_weather_all:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)"   ] || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.downloads.weather_forecast $(START) $(END) --all

forecast_next_all:
	@$(PY) -m src.downloads.forecast_next_all

# ------------------------------------------------------------
# Modeling & Prediction (GAM-sklearn)
# ------------------------------------------------------------
train_gam_skl:
	@$(PY) -m src.train_gam_skl

# Write 10-day CSVs (hourly_*.csv, peaks_*.csv) into $(PRED_DIR)
forecast_10d_gam:
	@ [ -n "$(START)" ] || (echo "START=YYYY-MM-DD required" && exit 2)
	@ [ -n "$(END)"   ] || (echo "END=YYYY-MM-DD required" && exit 2)
	$(PY) -m src.gam_predict $(START) $(END)

# One-line CSV for tomorrow (uses your print_predictions script)
predictions:
	@$(PY) -m src.print_predictions

# Full daily pipeline for the (START..END) window:
#   1) fetch forecasts for ALL zones
#   2) run GAM predictions (hourly + peaks)
#   3) print tomorrow’s single-line CSV to stdout
daily_10day:
	@echo "[daily] START=$(START) END=$(END)"
	$(PY) -m src.downloads.weather_forecast $(START) $(END) --all
	$(PY) -m src.gam_predict $(START) $(END)
	$(PY) -m src.print_predictions