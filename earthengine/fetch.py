import numpy as np

from init import *

fire = True
wind = True

if not wind and not fire:
    print("Nothing fetched!")

if fire:
    fire_dfs = multiple_date_fetch(dataset="NOAA/GOES/17/FDCF", band=["Power", "Mask"], resolution_km=2, date_ranges=date_ranges)
    for i, fire_df in enumerate(fire_dfs):
        time_str = date_ranges[i][0][5:].replace("-", "_", 1)
        fire_df.to_csv(f"datasets/fire_df_{time_str}.csv", index=False)

if wind:
    bands = ["u_component_of_wind_10m", "v_component_of_wind_10m", "temperature_2m",
             "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]

    wind_dfs = multiple_date_fetch(dataset="ECMWF/ERA5_LAND/HOURLY", resolution_km=2, band=bands, date_ranges=date_ranges)

    for i, wind_df in enumerate(wind_dfs):
        time_str = date_ranges[i][0][5:].replace("-", "_", 1)
        # wind_df = wind_df[12::24]   # 2pm every day

        wind_df["temperature_2m"] -= 273.15    # K to ˚C
        wind_df["speed"] = np.sqrt(wind_df["u_component_of_wind_10m"]**2 + wind_df["v_component_of_wind_10m"]**2)
        wind_df.drop(columns=["u_component_of_wind_10m", "v_component_of_wind_10m"], inplace=True)
        wind_df.to_csv(f"datasets/wind_df_{time_str}.csv", index=False)
