import numpy as np
import pandas as pd


data = pd.DataFrame([[1, 2, 3], [4, np.nan, 6], [7, np.nan, np.nan]])

print(data)

data = data.dropna()

print(data)
# --------------------------

data = pd.DataFrame([[1, 2, 3], [4, np.nan, 6], [7, np.nan, np.nan]])

data = data.fillna({0: data[0].mean(), 1: data[1].mean(), 2: data[2].mean()})

print(data)
# --------------------------
data = {
    'name': ['Michael', 'Steve', 'Liza', 'Jhon', 'Liza', 'Jhon'],
    'country': ['Canada', 'USA', 'Australia', 'Denmark', 'Australia', 'Denmark'],
    'age': [25, 32, 19, 23, 19, 23]
}

employees = pd.DataFrame(data)

employees = employees.drop_duplicates()
print(employees)
# --------------------------

# --------------------------

# --------------------------

