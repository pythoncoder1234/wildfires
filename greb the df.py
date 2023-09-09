import pandas as pd

df = pd.read_csv("wind_df.csv")
df = df[df["datetime"].apply(lambda val: "11:00:00" in val)]
df.to_csv("wind_df_greb.csv", index=False)
print(df)
