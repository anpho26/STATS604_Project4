# src/pipelines/daily_10day.py
from datetime import date, timedelta
import sys
from src.downloads.weather_forecast import main as wx_main
from src.forecast10days import main as fc_main

def main():
    start = date.today() + timedelta(days=1)
    end   = start + timedelta(days=9)

    # 1) fetch weather forecast for all zones
    sys.argv = ["weather_forecast", start.isoformat(), end.isoformat(), "--all"]
    wx_main()

    # 2) run the LightGBM models to produce hourly + PH/PD files
    sys.argv = ["forecast_10day", start.isoformat(), end.isoformat()]
    fc_main()

if __name__ == "__main__":
    main()