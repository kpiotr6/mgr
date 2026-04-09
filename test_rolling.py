import pandas as pd
import numpy as np

df = pd.DataFrame({'val': np.random.randn(100), 'is_at_edge': False})
df['time'] = pd.date_range('2023-01-01', periods=100)

INPUT_CHUNK_LENGTH = 10
col = 'val'

x = np.arange(len(df))
x_series = pd.Series(x, index=df.index)

cov_xy = df[col].rolling(window=INPUT_CHUNK_LENGTH, min_periods=2).cov(x_series)
var_x = x_series.rolling(window=INPUT_CHUNK_LENGTH, min_periods=2).var()
df[f"{col}_trend"] = (cov_xy / var_x).fillna(0)

print(df.head(15))
