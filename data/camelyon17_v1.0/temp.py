import pandas as pd

df = pd.read_csv("metadata.csv")
df.to_csv("metadata.csv")
#df['index_column'] = range(1, len(df) + 1)

