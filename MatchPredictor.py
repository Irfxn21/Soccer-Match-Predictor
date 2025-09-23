import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)

matches["date"] = pd.to_datetime(matches["date"])