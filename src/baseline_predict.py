import pandas as pd
import numpy as np

MODEL = "data/models/zone_hour_means_29.csv"
ZONES = [
    "AECO","AEPAPT","AEPIMP","AEPKPT","AEPOPT","AP","BC","CE","DAY","DEOK",
    "DOM","DPLCO","DUQ","EASTON","EKPC","JC","ME","OE","OVEC","PAPWR","PE",
    "PEPCO","PLCO","PN","PS","RECO","SMECO","UGI","VMEU",
]

def load_means(path=MODEL):
    m = pd.read_csv(path)
    # full grid; any missing hours → 0
    grid = (pd.MultiIndex.from_product([ZONES, range(24)], names=["load_area","hour"])
            .to_frame(index=False)
            .merge(m, how="left", on=["load_area","hour"])
            .fillna({"mean_mw": 0.0})
            .sort_values(["load_area","hour"]))
    return grid

def predict_all():
    grid = load_means()
    loads = {
        z: [int(round(x)) for x in grid.loc[grid.load_area.eq(z), "mean_mw"].values]
        for z in ZONES
    }
    peak_hour = {z: int(np.argmax(loads[z])) for z in ZONES}
    peak_day = {z: 0 for z in ZONES}  # simple baseline
    return loads, peak_hour, peak_day

if __name__ == "__main__":
    L, PH, PD = predict_all()
    print("Example:", "AECO", L["AECO"][:6], "PH=", PH["AECO"], "PD=", PD["AECO"])