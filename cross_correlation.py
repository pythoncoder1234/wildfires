from preprocessing import *
from timeinterp import save

norm_grids = []

for lag in range(0, 120, 10):
    corr_grids = []
    wind_norm_sqd = np.full((len(change_grids[0][1]), len(change_grids[0][1][0])), 0, dtype=np.float64)
    fire_norm_sqd = np.full((len(change_grids[0][1]), len(change_grids[0][1][0])), 0, dtype=np.float64)
    grid1 = wind1 = corr1 = None
    wind_gen = get_wind_interp(fire_coords)

    try:
        print("Lag:", lag)

        for i, grid in enumerate(sorted(fire_data.items())[lag:]):
            time, grid = grid

            grid = np.nan_to_num(grid)
            grid: np.ndarray

            fire_norm_sqd += grid * grid

            lats, lons, wind_data = next(wind_gen)

            # if i == 0:  # skip first
            #     lats, lons, wind_data = next(wind_gen)

            wind_norm_sqd += wind_data * wind_data

            corr_grid = grid * wind_data
            corr_grids.append(corr_grid)

            """with open(f"correlation/corr_{time.strftime('%H%M')}.csv", "w") as f:
                save(corr_grid, lons, lats, f)"""

    except StopIteration:
        print("StopIteration on wind data")

    sum_grid = sum(corr_grids)
    divide_grid = np.sqrt(wind_norm_sqd) * np.sqrt(fire_norm_sqd)
    norm_grid = np.nan_to_num(sum_grid / divide_grid)       # nan = divide by 0 (no correlation)
    norm_grids.append(norm_grid)

    with open(f"lag/lag_{lag}.csv", "w") as f:
        save(norm_grid, lons, lats, f)
