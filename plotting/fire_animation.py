import os

import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

data = []
pos = []
times = []
fire_df = pandas.read_csv("~/Downloads/fire_df.csv")
uniqueCount = 144

for i in range(uniqueCount):
    fire_no_nan = fire_df.iloc[i::uniqueCount]
    fire_no_nan = fire_no_nan[fire_no_nan.Power.notnull()]

    if i == 0:
        print(fire_no_nan[:3], "\nLength:", len(fire_no_nan))
        continue

    frameData = []
    pairs = []
    for tup in fire_no_nan.itertuples(index=False):
        times.append(tup[0])
        pairs.append(tup[2:4])
        frameData.append(tup[1])

    data.append(frameData)
    pos.append(pairs)

offset = 0.1
fig = plt.figure(figsize=(8, 7))
fig.subplots_adjust(hspace=1.5, wspace=1.5)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(fire_df["longitude"].min() - offset, fire_df["longitude"].max() + offset)
ax.set_ylim(fire_df["latitude"].min() - offset, fire_df["latitude"].max() + offset)
ax.set_title("Fire Grid")
times = np.unique(times)


def update(frame):
    try:
        scatter.set_offsets(pos[frame])
        scatter.set_array(data[frame])
        ax.set_title(f"Dixie Fire ({times[frame]})")
        if frame == uniqueCount - 2:
            plt.pause(3)

        return scatter,

    except Exception as e:
        print(e)


scatter = ax.scatter([], [], s=10,  c=[], cmap=plt.get_cmap("autumn_r"))
ani = FuncAnimation(fig, update, frames=uniqueCount-1, interval=200, blit=False)
# ani.save("plotting/animation.gif")
plt.show()
