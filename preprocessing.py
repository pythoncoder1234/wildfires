from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


DATA_PATH = "fire_df_08_05.csv"

def plot_fire(fire_data: list[tuple]):
    xs = []
    ys = []
    colors = []
    max_val = np.max(fire_data)

    for x, row in enumerate(fire_data):
        for y, val in enumerate(row):
            if val:
                xs.append(x)
                ys.append(y)
                colors.append([val / max_val, 0, 0])

    plt.scatter(xs, ys, marker="o", c=colors)
    plt.show()
    exit()


def get_fire_data() -> tuple[dict, tuple[float, float, float, float]]:
    with open("border.csv") as f:
        lat_min, lon_min, lat_max, lon_max, lat_dist, lon_dist = list(map(float, f.read().split(",")))

    def lat_i(lat):
        return round((lat - lat_min) / lat_dist)

    def lon_i(lon):
        return round((lon - lon_min) / lon_dist)

    width = lat_i(lat_max) + 1
    length = lon_i(lon_max) + 1

    print(f"Grid shape: {width}x{length}")

    grid = [[None for _ in range(length)] for _ in range(width)]

    time_grids = {}
    ula = set()
    ulo = set()

    for i, line in enumerate(open(DATA_PATH)):
        if i == 0:
            continue

        values = line.split(",")
        if values[1] == "":
            values[1] = 0

        time = pd.to_datetime(values[0])
        nums = list(map(float, values[1:]))
        v, h = lat_i(nums[3]), lon_i(nums[2])

        ula.add(nums[3])
        ulo.add(nums[2])

        time_values: list[tuple]
        try:
            time_grid = time_grids[time]
            time_grid[h][v] = nums[0]
        except KeyError:
            time_grid = np.full((len(grid[0]), len(grid)), np.nan, dtype=np.float64)
            time_grid[h][v] = nums[0]
            time_grids[time] = time_grid

    # print(sorted(ula), sorted(ulo), sep="\n")
    # print(len(ula), len(ulo))

    """
    for time, grid in time_grids.items():
        with open(f"fire_grid/fire_grid_{time.strftime('%H%M')}.csv", "w") as f:
            save(grid, np.arange(lon_min, lon_max + lon_dist, lon_dist),
                 np.arange(lat_min, lat_max + lat_dist, lat_dist), f)
    """

    return time_grids, (lat_min, lon_min, lat_max, lon_max)


def get_wind_interp(fire_coords):
    folder_path = Path("wind_interp")

    lat_ind = lon_ind = 69420
    lat_min, lon_min, lat_max, lon_max = fire_coords
    nums = []
    lats = nums[lat_ind:]

    for fileNum, file in enumerate(sorted(folder_path.iterdir())):
        lons = []
        grid = []

        for i, line in enumerate(file.open()):
            nums = list(map(float, line.split(",")[i == 0:]))

            if i == fileNum == 0:
                for j, num in enumerate(nums):
                    if num >= lat_min:
                        lat_ind = j
                        break

                lats = nums[lat_ind:]
                continue

            if nums[0] < lon_min or nums[0] > lon_max:
                continue

            if lon_ind == 69420:
                lon_ind = i
                print(f"Cutoff: ({lat_ind}, {lon_ind})")

            lons.append(nums[0])
            grid.append(nums[lat_ind + 1:])

        grid = np.array(grid)

        yield lats, lons, grid


change_grids = None
fire_data = fire_coords = None

resize_wind = True
resize_threshold = 0.01

def preprocess(data_path=DATA_PATH):
    global fire_data, fire_coords, change_grids

    fire_data, fire_coords = get_fire_data()

    prev = None
    change_grids = []

    i = 0
    for time, data in sorted(fire_data.items()):
        if time.hour >= 23 and time.minute > 0:
            break

        if prev is None:
            prev = data
            continue

        i += 1

        if i == 19:
            pass

        change = data - prev

        change_grids.append((time, change))

        prev = data


if __name__ == '__main__':
    preprocess()
