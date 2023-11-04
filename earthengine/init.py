import ee

import pandas as pd

ee.Initialize()

km_lon_deg = 110.574  # Kilometers for 1 degree of longitude
km_lat_deg = 111.321  # Kilometers for 1 degree of latitude
date_range = ["2021-08-01", "2021-08-02"]
date_ranges = [[f"2021-08-{i:02}", f"2021-08-{i+1:02}"] for i in range(1, 11)]
fire_name = "Dixie"

data = {
    "Dixie": {
        "coordinates": [
            -121.5, 40.5,
            -120.5, 40.5,
            -120.5, 39.5,
            -121.5, 39.5
        ]
    }
}


def multiple_date_fetch(dataset, band, resolution_km, date_ranges) -> list[pd.DataFrame]:
    out = []
    for i, dr in enumerate(date_ranges):
        print(f"Fetching date range {date_ranges[i]}")
        out.append(get_dataset(dataset, band, resolution_km, dr))
    return out


def get_dataset(dataset, band, resolution_km, dr):
    print(f"Fetching '{dataset}'...")
    fire = data[fire_name]

    region = ee.Geometry.Polygon(fire["coordinates"])
    collection = ee.ImageCollection(dataset)
    collection = collection.select(band).filterDate(*dr[:2])

    if len(dr) > 2:
        collection = collection.filter(ee.Filter.calendarRange(dr[2], dr[3], "HOUR"))

    collection = collection.getRegion(region, resolution_km * 1000).getInfo()

    return ee_to_df(collection, band)


def ee_to_df(arr, band):
    """Converts an ee.List to a Pandas.DataFrame."""

    df = pd.DataFrame(arr)

    if type(band) is str:
        bands = [band]
    else:
        bands = band

    headers = df.iloc[0]

    df = pd.DataFrame(df.values[1:], columns=headers)

    for b in bands:
        df[b] = pd.to_numeric(df[b])

    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    df = df[['datetime', *bands, 'longitude', 'latitude']]

    print("Done!")

    return df
