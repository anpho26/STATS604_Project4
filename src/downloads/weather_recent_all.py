from datetime import date, timedelta
import os, sys
from src.downloads.weather_history import main as run

def main():
    days = int(os.getenv("DAYS", "31"))  # default 31
    end = date.today()
    start = end - timedelta(days=days)
    sys.argv = ["weather_history", start.isoformat(), end.isoformat(), "--all"]
    run()

if __name__ == "__main__":
    main()