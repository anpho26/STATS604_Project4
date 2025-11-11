# STATS 604 — Power-Grid Forecast

End-to-end pipeline to forecast hourly PJM loads for 29 zones, plus:

- Peak hour per zone/day
  
- Peak day (top-2 days per zone over the 10-day window)

The project ships with raw PJM power data and weather history. For daily inference it fetches Open-Meteo forecasts, runs LightGBM models, and prints the one-line CSV in the required format.

## Contents
```
.
├─ data/
│  ├─ raw/
│  │  ├─ power/                 # PJM hrl_load_metered_*.csv (provided)
│  │  ├─ weather/               # Meteostat history per zone (provided)
│  │  ├─ weather_forecast/      # Open-Meteo cache (created/updated)
│  │  └─ zones_locations.csv    # zone → lat/lon
│  ├─ models/                   # trained models & baseline artifacts
│  └─ predictions_10day/        # model outputs (created)
├─ src/
│  ├─ downloads/
│  │  ├─ weather_history.py     # Meteostat hourly history per zone
│  │  ├─ weather_forecast.py    # Open-Meteo hourly forecast per zone
│  │  ├─ forecast_next_all.py   # tomorrow..+9 helper
│  │  └─ weather_recent_all.py  # last N days helper
│  ├─ train_lgbm.py             # train per-zone LightGBM regressors
│  ├─ forecast10days.py         # run models → hourly + peak files
│  ├─ pipelines/daily_10day.py  # fetch forecast + run 10-day forecast
│  ├─ baseline_predict.py       # simple historical-mean baseline
│  └─ print_predictions.py      # prints required one-line CSV (uses LGBM)
├─ Makefile
└─ requirements.txt
```

Zone order (fixed):
AECO, AEPAPT, AEPIMP, AEPKPT, AEPOPT, AP, BC, CE, DAY, DEOK, DOM, DPLCO, DUQ, EASTON, EKPC, JC, ME, OE, OVEC, PAPWR, PE, PEPCO, PLCO, PN, PS, RECO, SMECO, UGI, VMEU

## Quick start (local)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make             # trains LightGBM models
make daily_10day # (optional) fetches forecast tomorrow..+9 and runs the 10-day prediction
make predictions # prints the required one-line CSV for TOMORROW (ET)
```

### Useful Make targets
```
make                     # runs all analyses (no downloads of raw data)
make predictions         # prints "YYYY-MM-DD, L1_00,...,L29_23, PH_1..PH_29, PD_1..PD_29"
make train_lgbm          # trains per-zone LGBM models from raw power + weather history
make daily_10day         # fetch Open-Meteo (tomorrow..+9) and run predictions
make forecast_weather_all START=YYYY-MM-DD END=YYYY-MM-DD
make forecast_10d        START=YYYY-MM-DD END=YYYY-MM-DD   # run models for an explicit window
make weather_recent_all  # refresh last N (default 31) days of Meteostat history
make rawdata             # delete & re-download weather raw (keeps power raw)
make clean               # delete derived artifacts (keeps raw data and code)
```

### Notes

-	The forecast window uses Open-Meteo’s horizon (≈16 days ahead).
-	Timezone: the printed header date is tomorrow in US/Eastern, and hourly times are EPT in outputs.
  
## Docker

A multi-arch image is published: 
```
anpho26/stats604_proj4:latest
```

### Required run models
Interactive shell:
```
docker run -it --rm anpho26/stats604_proj4:latest
# inside container:
make help
```

One-liner predictions (prints and exits):
```
docker run -it --rm anpho26/stats604_proj4:latest make predictions
```

The image contains:

- code in /app/src
- raw data in /app/data/raw/{power,weather,weather_forecast}
- trained models in /app/data/models

## Model summary

### Features (per zone):
- calendar features: hour of day, day of week, month, holiday flags (if available)
- lagged loads: e.g., 24h, 48h, 168h (weekly) where present in training
- weather features: temp (temp), humidity (rhum), precipitation (prcp), windspeed (wspd), pressure (pres), weather code (coco)
- Model: LightGBM regressor per zone; trained on 2021 → most recent history shipped.

### Outputs
- data/predictions_10day/hourly_<START>_<END>.csv
- columns: zone,time_ept,mw_pred
- data/predictions_10day/peaks_<START>_<END>.csv
- columns: zone,date,peak_hour,peak_load,peak_day_flag

Peak hour is argmax over the 24 hourly predictions.

Peak day flag marks the top-2 days by peak load per zone across the 10-day window.

### Data sources

- PJM hrl_load_metered.csv (provided in data/raw/power)
- Meteostat hourly history (python package meteostat)
- Open-Meteo hourly forecast (https://open-meteo.com/)

data/raw/zones_locations.csv maps each zone to lat/lon used to choose stations / forecast points.
