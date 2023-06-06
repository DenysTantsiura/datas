import pandas as pd

date = pd.Timestamp('2023-09-10')

print(date)  # 2023-09-10 00:00:00

# --------------------------
date = pd.to_datetime('2021-09-10 2:54:13')

print(date)  # 2021-09-10 2:54:1
# --------------------------
 
date = pd.date_range(start='2021-09-01', freq='D', periods=8)

temperature = pd.Series([23, 17, 17, 16, 15, 14, 17, 20], index=date)

print(temperature)
# --------------------------

# --------------------------

# --------------------------

# --------------------------

