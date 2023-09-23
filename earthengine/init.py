import ee

import pandas as pd

ee.Initialize()

km_lon_deg = 110.574  # Kilometers for 1 degree of longitude
km_lat_deg = 111.321  # Kilometers for 1 degree of latitude
date_range = ["2021-08-05", "2021-08-06", 0, 3]
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


def get_dataset(dataset, band, resolution_km):
    print(f"Fetching '{dataset}'...")
    fire = data[fire_name]

    region = ee.Geometry.Polygon(fire["coordinates"])
    collection = ee.ImageCollection(dataset)
    collection = collection.select(band).filterDate(*date_range[:2])

    if len(date_range) > 2:
        collection = collection.filter(ee.Filter.calendarRange(date_range[2], date_range[3], "HOUR"))

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
