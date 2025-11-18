from datetime import date, timedelta
import sys
from src.downloads.weather_forecast import main as wx_main

def main():
    start = date.today() + timedelta(days=1)
    end   = start + timedelta(days=9)
    sys.argv = ["weather_forecast", start.isoformat(), end.isoformat(), "--all"]
    wx_main()

if __name__ == "__main__":
    main()