from pathlib import Path

import pandas as pd


AIS_FILE = Path(__file__).resolve().parent.parent / "database" / "ais.csv"


def load_ais_data():
    df = pd.read_csv(AIS_FILE)

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

    return df


def get_vessels_from_ais_csv():
    df = load_ais_data()

    latest = (
        df.sort_values("BaseDateTime")
        .groupby("MMSI")
        .tail(1)
    )

    vessels = latest[
        [
            "MMSI",
            "VesselName",
            "LAT",
            "LON",
            "SOG",
            "COG",
            "Heading",
        ]
    ]

    latest = latest.astype(object).where(
        pd.notna(latest),
        None
    )

    vessels = latest.to_dict(orient="records")

    for vessel in vessels:
        if vessel["VesselName"] is None:
            vessel["VesselName"] = "UnIdentified"

    return vessels

def get_trajectories_from_ais_csv():

    df = load_ais_data()

    trajectories = {}

    for mmsi, vessel_data in df.groupby("MMSI"):

        vessel_data = vessel_data.sort_values("BaseDateTime")

        points = vessel_data[
            ["LAT", "LON"]
        ].values.tolist()

        trajectories[str(mmsi)] = points

    return trajectories