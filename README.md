# STATS 604 — Power-Grid Forecast

End-to-end pipeline to forecast hourly PJM loads for 29 zones, plus:

- Peak hour per zone/day
  
- Peak day (top-2 days per zone over the 10-day window)

The project ships with raw PJM power data and weather history. For daily inference it fetches Open-Meteo forecasts, runs GAM-like models, and prints the one-line CSV in the required format.

## Contents
```
.
├─ data/
│  ├─ raw/
│  │  ├─ power/                       # PJM hourly load CSVs (OSF/instructor bundle)
│  │  ├─ weather/                     # Meteostat history per zone (created)
│  │  ├─ weather_forecast/            # Open-Meteo hourly forecast cache (created)
│  │  └─ zones_locations.csv          # zone → lat/lon used by weather scrapers
│  ├─ models/                         # trained per-zone Ridge (“GAM-skl”) models:
│  │                                   #   gam_skl_<ZONE>.joblib + gam_skl_<ZONE>.meta.json
│  ├─ predictions_10day/              # model outputs (hourly_*, peaks_*, one_line_*.csv)
│  └─ figures/                        # plots exported by src/plots_gam.py
├─ src/
│  ├─ downloads/
│  │  ├─ pjm_osf_download.py          # download & unpack 10-yr PJM bundle from OSF
│  │  ├─ weather_history.py           # Meteostat hourly history per zone (2021→today)
│  │  ├─ weather_forecast.py          # Open-Meteo hourly forecast per zone for a window
│  │  ├─ weather_recent_all.py        # helper: refresh last N days for all zones
│  │  └─ forecast_next_all.py         # helper: fetch forecast for tomorrow..+9 for all zones
│  ├─ constants.py                    # zone list and shared paths
│  ├─ train_gam_skl.py                # train per-zone Ridge on residualized log target
│  ├─ gam_predict.py                  # make 10-day forecasts → hourly_*.csv + peaks_*.csv
│  ├─ print_predictions.py            # emit required one-line CSV (uses GAM-skl models)
│  └─ plots_gam.py                    # quick visual comparisons (2025 vs past years)
├─ Makefile                           # convenience targets (train_gam_skl, forecast_10d_gam, predictions, etc.)
├─ dockerfile                         # reproducible environment (python:3.11-slim)
└─ requirements.txt                   # Python dependencies
```

Zone order (fixed):
AECO, AEPAPT, AEPIMP, AEPKPT, AEPOPT, AP, BC, CE, DAY, DEOK, DOM, DPLCO, DUQ, EASTON, EKPC, JC, ME, OE, OVEC, PAPWR, PE, PEPCO, PLCO, PN, PS, RECO, SMECO, UGI, VMEU

## Quick start (local)

```
# Create/activate a venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) fetch the PJM bundle to data/raw/power/
python -m src.downloads.pjm_osf_download data/raw/power

# Train per-zone Ridge models on residualized log target
make train_gam_skl

# Option A: write 10-day CSVs (hourly_* and peaks_*) for an explicit window
make forecast_10d_gam START=2025-11-20 END=2025-11-29

# Option B: print the one-line submission for TOMORROW (US/Eastern)
# Peak-day flags are computed over the fixed window set in the Makefile.
make predictions
```

### Useful Make targets
```
make                       # (default) see Makefile 'all' or run `make help` if present

make predictions           # prints "YYYY-MM-DD, L1_00,...,L29_23, PH_1..PH_29, PD_1..PD_29"
                           # uses FIXED_START / FIXED_END (set in Makefile or override on CLI)

make train_gam_skl         # trains per-zone Ridge (“GAM-skl”) models from power + weather

make forecast_10d_gam START=YYYY-MM-DD END=YYYY-MM-DD
                           # runs models and writes data/predictions_10day/{hourly_*,peaks_*}.csv

make forecast_weather_all START=YYYY-MM-DD END=YYYY-MM-DD
                           # fetch Open-Meteo forecast for all zones for the window

make weather_recent_all    # refresh last N (default 31) days of Meteostat history

make rawdata               # delete & re-download weather raw (keeps power raw)

make clean                 # delete derived artifacts (models, predictions, processed); keeps r
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

### Features & target (per zone)

**Target (residualized log load).**  
We model \(y^*_{z,t} = \log(1+L_{z,t}) - \tilde b_{z,\mathrm{hour}(t)}\),  
where \(\tilde b\) is the fixed same-hour log baseline computed at the train cutoff.  
At inference we add a small bias correction \(\delta_z\) (mean residual over the last 14 days).

**Calendar**
- Fourier hour-of-day terms (period 24; K=3 sine/cosine pairs).
- Fourier day-of-year terms (period 365.25; K=2 pairs).
- Day-of-week dummies (Mon–Sat) + weekend flag.
- Holiday flags (Thanksgiving, Black Friday, etc.).
- Year dummies (2022, 2023, 2025; 2024 is baseline).

**Weather**
- **Daily mean temperature** (`temp_day_mean`) computed from the hourly forecast and broadcast to hours.  
  *(Intentionally simple/robust; no humidity/wind/etc.)*

**Model**
- One **Ridge regression** (L2-regularized linear model) per zone with standardized features.  
- \(\alpha\) selected by RMSE on the held-out 2024 Thanksgiving window, then refit and used for 2025 predictions.
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
