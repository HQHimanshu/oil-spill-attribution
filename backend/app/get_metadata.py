from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
path = BASE_DIR / "database" / "synthetic_oil_spill_dataset_20" / "metadata.csv"

def get_metadata_for_image(image_id):
    df = pd.read_csv(path)

    result = df[df["id"] == image_id]

    if result.empty:
        return None

    row = result.iloc[0]

    return {
        "id": row["id"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "acquisition_date": row["acquisition_date"],
        "sensor": row["sensor"],
        "spill_class": row["spill_class"],
        "coordinate_status": row["coordinate_status"]
    }