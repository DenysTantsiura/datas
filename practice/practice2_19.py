import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

grid = np.arange(-1, 1, 0.5)  # (-10, 10, 0.5)
x, y = np.meshgrid(grid, grid)
z = x ** 2 * y ** 2 + 2

ax.plot_wireframe(x, y, z)

# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

grid2 = np.arange(-2, 2, 1)
x, y = np.meshgrid(grid, grid2)
# z = x ** 2 * y ** 2 + 2
z = np.array([np.arange(-2, 2, 1), np.arange(-2, 2, 1), np.arange(-2, 2, 1), np.arange(-2, 2, 1)])

ax.plot_wireframe(x, y, z)

xl = np.array(['a', 'b', 'c', 'd'])  # np.array(p1t1m1['регіон'])
plt.xticks(ticks=grid2, labels=xl, rotation=85)

plt.show()

# -- surface 
