import numpy as np

from init import *

date_range = ["2021-08-01", "2021-08-02"]
time_str = date_range[0][5:].replace("-", "_", 1)
fire = False
wind = False

print(time_str)

if not wind and not fire:
    print("Nothing fetched!")

if fire:
    fire_df = get_dataset(dataset="NOAA/GOES/17/FDCF", band=["Power", "Mask"], resolution_km=2)
    fire_df.dropna(inplace=True)
    fire_df.to_csv(f"fire_df_{time_str}.csv", index=False)
    print(fire_df.dropna())

if wind:
    bands = ["u_component_of_wind_10m", "v_component_of_wind_10m", "temperature_2m",
             "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]

    wind_df = get_dataset(dataset="ECMWF/ERA5_LAND/HOURLY", resolution_km=2, band=bands)
    # wind_df = wind_df[12::24]   # 2pm every day

    wind_df["temperature_2m"] -= 273.15    # K to ˚C
    wind_df["speed"] = np.sqrt(wind_df["u_component_of_wind_10m"]**2 + wind_df["v_component_of_wind_10m"]**2)
    wind_df.drop(columns=["u_component_of_wind_10m", "v_component_of_wind_10m"], inplace=True)
    wind_df.to_csv(f"wind_df_{time_str}.csv", index=False)
    print(wind_df)
