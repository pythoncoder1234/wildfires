from matplotlib import cm

from cross_correlation import *

lat_mesh, lon_mesh = np.meshgrid(lats, lons)
coords = []

fig = plt.figure(figsize=(16, 8))

for i in range(0, 100, 10):
    ax = fig.add_subplot(2, 5, i // 10 + 1, projection='3d')
    grid = norm_grids[i // 10]

    if i == 0:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    lag_str = ""
    m = i * 10

    if m // 60 > 0:
        lag_str += f"{m // 60}h"

    if m % 60 > 0 or m // 60 == 0:
        if m // 60 == 0:
            lag_str = f"{m % 60}m"
        else:
            lag_str += f" {m % 60}m"

    max_val = abs(grid).max()
    if max_val != grid.max():
        max_val = -max_val

    ax.set_title(f"Lag: {lag_str}   Max: {round(max_val, 2)}")
    plot = ax.plot_surface(lat_mesh, lon_mesh, grid, cmap=cm.gist_earth, vmin=0, vmax=1)

plt.savefig(f"plotting/corr_coefs.png", bbox_inches="tight", pad_inches=0.5)
# plt.show()
