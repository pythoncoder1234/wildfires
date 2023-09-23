import pandas as pd
from time import time

fire = pd.read_csv("fire_df_08_01.csv")
wind = pd.read_csv("wind_df_08_01.csv")

def interpolate(index):
    start = time()
    interp = wind.copy()
    prev_row = None
    for row in interp.itertuples():
        if prev_row:
            prev = interp.iloc[row[0]-1]
            interp.iloc[row[0]-1, 0] = prev[0].replace(minute=index * 10)
            interp.iloc[row[0]-1, 1:] += (row[1:] - prev) * (index / 6)

        prev_row = row

    print(f"Took {round(time() - start, 3)}s")
    return interp

# interpolated = pd.concat([interpolate(time) for time in range(1, 6)])

for df in [fire, wind]:
    df['datetime'] = df['datetime'].apply(lambda time: pd.to_datetime(time).replace(second=0, microsecond=0))
    df[['longitude', 'latitude']] = df[['longitude', 'latitude']].round(5)

merge = pd.merge(fire, wind, how="outer", on=["datetime", "longitude", "latitude"])
# merge.to_csv("merge.csv", index=False)
