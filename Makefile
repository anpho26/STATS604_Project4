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

# -------- configurable knobs --------
# Zone used for quick plots / notebook (you can override: `make ZONE=PEPCO`)
ZONE ?= AECO

# Notebook path + rendered output
NB       := overview.ipynb
NB_HTML  := data/figures/overview$(ZONE).html

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
all: train_gam_skl predictions plots notebook
	@echo "[all] done."

help:
	@echo "Usage:"
	@echo "make                           # train + predictions + figures + notebook"
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
	@$(PY) -m src.gam_predict $(START) $(END)

# Fixed 10-day window you want to use for peak-day flags
FIXED_START := 2025-11-20
FIXED_END   := 2025-11-29

# Still prints the one-line CSV for **tomorrow**,
# but peak-day flag is computed from the fixed window.
predictions:
	@$(PY) -m src.print_predictions --fixed-start=$(FIXED_START) --fixed-end=$(FIXED_END)

# Full daily pipeline for the (START..END) window:
#   1) fetch forecasts for ALL zones
#   2) run GAM predictions (hourly + peaks)
#   3) print tomorrow’s single-line CSV to stdout
daily_10day:
	@echo "[daily] START=$(START) END=$(END)"
	$(PY) -m src.downloads.weather_forecast $(START) $(END) --all
	$(PY) -m src.gam_predict $(START) $(END)
	$(PY) -m src.print_predictions



# ===== Quick plots for ONE zone (uses src/plots_gam.py) =====
plots: figures
	$(PY) -m src.plots_gam $(ZONE)

figures:
	mkdir -p data/figures

# ===== Execute EDA notebook =====
# Preferred: papermill (parameterize ZONE). Fallback: nbconvert.
notebook: figures
	@echo "[nb] executing $(NB) for ZONE=$(ZONE)…"
	@if command -v papermill >/dev/null 2>&1; then \
	  papermill "$(NB)" "$(NB_HTML)" -p ZONE "$(ZONE)"; \
	  echo "[nb] wrote $(NB_HTML)"; \
	else \
	  echo "[nb] papermill not found; using nbconvert with notebook's default ZONE"; \
	  jupyter nbconvert --to html --execute "$(NB)" --ExecutePreprocessor.kernel_name=python3 --output-dir=data/figures; \
	fi