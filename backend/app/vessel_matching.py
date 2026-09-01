import math


def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1

    dlon = (
        math.radians(lon2)
        -
        math.radians(lon1)
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


def calculate_correlation(
    distance_km
):

    if distance_km <= 10:
        return 100

    elif distance_km <= 30:
        return 80

    elif distance_km <= 60:
        return 60

    elif distance_km <= 100:
        return 40

    elif distance_km <= 150:
        return 20

    else:
        return 0


def rank_vessels(
    spill_latitude,
    spill_longitude,
    vessels
):

    results = []


    for vessel in vessels:

        distance = calculate_distance(

            spill_latitude,
            spill_longitude,

            vessel["LAT"],
            vessel["LON"]

        )


        score = calculate_correlation(
            distance
        )


        if score >= 80:

            risk = "HIGH"

        elif score >= 40:

            risk = "MEDIUM"

        else:

            risk = "LOW"


        results.append({

            "mmsi":
                vessel["MMSI"],

            "name":
                vessel["VesselName"],

            "latitude":
                vessel["LAT"],

            "longitude":
                vessel["LON"],

            "distance_km":
                round(distance, 2),

            "correlation_score":
                score,

            "risk_level":
                risk

        })


    results.sort(

        key=lambda x:
        x["correlation_score"],

        reverse=True

    )


    return results