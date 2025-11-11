# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# make for targets, libgomp for LightGBM wheels, tzdata for time-handling
RUN apt-get update && apt-get install -y --no-install-recommends \
      make libgomp1 tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- code & make targets ---
COPY Makefile .
COPY src/ src/

# --- RAW DATA (exactly as you have locally) ---
# PJM power CSVs
COPY data/raw/power/              data/raw/power/
# Meteostat history you scraped
COPY data/raw/weather/            data/raw/weather/
# (optional) Open-Meteo forecast cache if you keep it
COPY data/raw/weather_forecast/   data/raw/weather_forecast/

# --- models/artifacts (optional but recommended to reproduce fast) ---
# If you have trained models, include them:
COPY data/models/                 data/models/

# Python deps
# If you already have requirements.txt in repo, use it; otherwise list here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# default run: interactive bash
CMD ["/bin/bash"]