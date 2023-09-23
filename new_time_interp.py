import pandas as pd

fire = pd.read_csv("fire_df_08_01.csv")
wind = pd.read_csv("wind_df_08_01.csv")

for df in [fire, wind]:
    df['datetime'] = df['datetime'].apply(lambda time: pd.to_datetime(time).replace(second=0, microsecond=0))
    df[['longitude', 'latitude']] = df[['longitude', 'latitude']].round(5)

merge = pd.merge_ordered(fire, wind, how="left", on=['datetime', 'longitude', 'latitude'], fill_method="ffill")
merge.to_csv("merge.csv", index=False)
