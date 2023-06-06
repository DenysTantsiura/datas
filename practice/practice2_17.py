import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

theta_max = 8 * np.pi
n = 1000
theta = np.linspace(0, theta_max, n)
x = theta
z = np.cos(theta)
y = np.sin(theta)
ax.plot(x, y, z, "g")

plt.show()

# -- simple 3d ----
