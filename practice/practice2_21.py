import numpy as np
import pandas as pd
import seaborn as sns


my_data = pd.read_csv('auto-mpg.csv')

data = sns.load_dataset('mpg')

print(my_data.head())
print('->'*30)
print(data.head())
my_data.columns = data.columns
print(my_data == data)
print(my_data is data)
print('-='*30)
my_data['origin'] = my_data['origin'].replace(to_replace=1, value='usa')
my_data['origin'] = my_data['origin'].replace(to_replace=2, value='europe')
# https://stackoverflow.com/questions/15891038/change-column-type-in-pandasv
print(type(my_data['horsepower'][0]))
my_data['horsepower'] = my_data['horsepower'].replace(to_replace=np.nan, value='0').replace(to_replace='?', value='0')
my_data = my_data.astype({'horsepower': int})

my_data['horsepower'] = my_data['horsepower'] / 1.0

print(my_data)
print(my_data.head() == data.head())
print(my_data is data)
