# import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


X = np.array([
    [1, 1],
    [1, 2],
    [2, 2],
    [2, 3]
])

y = np.dot(X, np.array([1, 2])) + 3

regressor = LinearRegression().fit(X, y)

print(regressor.predict(np.array([[3, 5]])))

df = pd.read_csv('practice/Housing.csv')


matplotlib.use('Qt5Agg')


def linear_regression_hypothesis(w_0: float, w_1: float, x: float) -> float:
    return w_0 + w_1 * x


def loss_function(w_0: float, w_1: float, df: pd.DataFrame) -> float:
    n = df.area.shape[0]  # df.shape[0]
    cost = 0
    for x, y in zip(df.area, df.price):
        cost += (y - linear_regression_hypothesis(w_0, w_1, x)) ** 2

    return cost/(2*n)


grid_w_0 = np.arange(-2000, 2000, 200)
grid_w_1 = np.arange(-10000, 10000, 1000)
w_0, w_1 = np.meshgrid(grid_w_0, grid_w_1)
z = loss_function(w_0, w_1, df)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(w_0, w_1, z)
plt.show()
