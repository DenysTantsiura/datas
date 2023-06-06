import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
date = pd.date_range(start="2021-09-01", freq="D", periods=8)
day = [23, 17, 17, 16, 15, 14, 17, 20]
night = [19, 11, 16, 11, 10, 10, 11, 16]
df = pd.DataFrame({'date':date, 'day_temperature': day, 'night_temperature': night})
sns.lineplot(data=df)


# sns.plot()

# seaborn ----
