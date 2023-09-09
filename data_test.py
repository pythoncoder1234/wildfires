from timeinterp import save
from variables import *

generator = get_file_data('wind_df_greb.csv')
time, lats, lons, grid = next(generator)
with open(f"interp_{time.strftime('%H%M')}.csv", "w") as f:
    save(grid, lats, lons, f)
