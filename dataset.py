import os
import re
from datetime import datetime

import pygrib


def fetch_data(band):
    data = ds.message(band)
    return data.data(lat1=39.5, lon1=-121.5, lat2=40.5, lon2=-120.5)


def parse_time(timestring):
    hour, minute = timestring.split(":")
    hour, minute = int(hour), int(minute)

    return (hour * 4 + minute // 15) * 2, datetime(2022, 12, 8, hour, minute)


PATH = "/noaa_wind_grb2/"
files = sorted([PATH + p for p in os.listdir(PATH)])
fileNum, time = parse_time("16:15")
print(files[fileNum], end="\n\n")

ds = pygrib.open(files[fileNum])

u, ulats, ulons = fetch_data(5)
v, vlats, vlons = fetch_data(6)


def plot():
    from matplotlib import pyplot as plt

    print("Total vector count:", len(ulats))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.quiver(ulons, ulats, u, v)
    plt.show()


# plot()


if __name__ == '__main__':
    for i, line in enumerate(ds):
        if i > 30:
            break

        print(str(line).replace(":", ": ").replace(" s**-1", "/s"))

    print(ulats)
    print(ulons)
