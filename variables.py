import os
from datetime import datetime
from glob import glob
from typing import Generator, Any

import numpy as np
import pandas as pd

URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/rtma2p5_ru.20230106/"
DATASET_FOLDER = "noaa_wind_grb2"

datasets = os.listdir(DATASET_FOLDER)

WIND = 5
TEMP = 0
LEAF_HIGH = 1
LEAF_LOW = 2

index = WIND


def clear_folder(folder, **kwargs):
    if input(f"Clear folder '{folder}'? (y/n): ").lower() != "y":
        return

    for file in glob(f"{folder}/*.{kwargs.get('file_ending', 'grb2')}"):
        os.remove(file)


def load_file(path):
    data = {}

    for i, line in enumerate(open(path)):
        if i == 0:
            continue

        kw = line.split(",")
        row = []

        for j, num in enumerate(kw):
            if j == 0:
                time = pd.to_datetime(num)
                continue

            if num:
                row.append(float(num))
            else:
                row.append(None)

        try:
            for j, num in enumerate(row):
                data[time][j].append(num)
        except KeyError:
            data[time] = list(map(lambda x: [x], row))

    return data


def get_file_data(path) -> Generator[tuple[datetime, list[float], list[float], np.ndarray], Any, None]:
    data = load_file(path)

    for k, v in data.items():
        for s in v:
            print(s[:5], "...", s[-5:])

        lats = v[4]
        lons = v[3]
        yield k, *construct_grid(lats, lons, v[index])


def construct_grid(lats, lons, data):
    def lat_i(lat):
        return round((lat - lat_min) / lat_dist)

    def lon_i(lon):
        return round((lon - lon_min) / lon_dist)

    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    unique_lats = unique_lats[~np.isnan(unique_lats)]
    unique_lons = unique_lons[~np.isnan(unique_lons)]
    lat_min = unique_lats[0]
    lon_min = unique_lons[0]
    lat_dist = unique_lats[1] - unique_lats[0]
    lon_dist = unique_lons[1] - unique_lons[0]
    lat_size = len(unique_lats)
    lon_size = len(unique_lons)

    print(f"Grid shape: {lat_size}x{lon_size}")

    grid = np.full((lon_size, lat_size), np.nan, dtype=np.float64)

    for i in range(len(lats)):
        first = lon_i(lons[i])
        second = lat_i(lats[i])

        grid[first][second] = data[i]

    return unique_lats, unique_lons, grid


print("Current working directory:", os.getcwd())
