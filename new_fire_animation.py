import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import normalize

def read_data():
    global times, wind, fire
    merge = pd.read_csv("merge.csv")
    merge = merge.sort_values(['longitude', 'latitude'])
    merge['datetime'] = pd.to_datetime(merge['datetime'])
    merge = merge.loc[merge['datetime'].apply(lambda value: value.minute % 10 == 0)]
    grouped = merge.groupby('datetime')
    times = list(map(str, grouped.groups.keys()))

    wind = grouped['speed'].apply(process).to_list()
    fire = grouped['Power'].apply(process).to_list()
    wind = normalize(wind) * 10
    fire = normalize(fire) * 10


def process(row):
    return np.pad(np.nan_to_num(row), (3080 - len(row), 0))

def update(frame):
    fig.suptitle(times[frame])
    arr = fire[frame].reshape(55, 56)
    arr[arr == 0] = np.nan
    img.set_data(arr)
    return img,

if not len(fire):
    read_data()

fire = fire[450:]
times = times[450:]
fig = plt.figure()
img = plt.matshow(fire[0].reshape(55, 56), fignum=0, vmin=0, vmax=10, cmap="autumn_r")
ani = FuncAnimation(fig, update, len(fire), interval=200, blit=False)
