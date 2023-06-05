import matplotlib.pyplot as plt
import numpy as np


theta = np.linspace(0, 2.0 * np.pi, 1000)

r = np.sin(6 * theta)
plt.polar(theta, r)
plt.show()

# -----rose
