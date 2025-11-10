from datetime import date
from src.baseline_predict import predict_all, ZONES

def main():
    loads, peak_hour, peak_day = predict_all()

    cells = [f"\"{date.today().isoformat()}\""]  # "YYYY-MM-DD"
    # L1_00..L1_23, L2_00.., ..., L29_23  (zone order = ZONES)
    for z in ZONES:
        cells.extend(str(int(x)) for x in loads[z])
    # PH_1..PH_29
    for z in ZONES:
        cells.append(str(int(peak_hour[z])))
    # PD_1..PD_29
    for z in ZONES:
        cells.append(str(int(peak_day[z])))

    print(", ".join(cells))

if __name__ == "__main__":
    main()