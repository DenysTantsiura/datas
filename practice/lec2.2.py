import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# date = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
# plt.figure(figsize=(5, 5))
# plt.plot(date, date ** 2, linestyle="-.", color='g', label = 'x**2')
# plt.plot(date, date ** 3, marker='^', label = 'x**3')
# plt.xlabel('x', fontsize='small', color='midnightblue')
# plt.ylabel('y', fontsize='small', color='midnightblue')
# plt.title('y = x ** 2 & x ** 3', fontsize=15)
# plt.text(4, 40, 'texts')
# plt.grid()
# plt.legend()

# # plt.show()
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(date, date ** 2)
# axs[1].plot(date, date ** 3)
# fig.suptitle("123")

# plt.figure(figsize=(6, 6))
# x = ['A', "B", 'C', "D"]
# y = [2, 3, 5, 1]
# plt.bar(x, y)
# plt.barh(x, y)


# labels = [
#     "Junior Software Engineer",
#     "Senior Software Engineer",
#     "Software Engineer",
#     "System Architect",
#     "Technical Lead",
# ]

# data = [63, 31, 100, 2, 11]
# explode = [0.15, 0, 0, 0, 0]
# plt.figure(figsize=(6, 6))
# plt.pie(
#     data,
#     labels=labels,
#     shadow=True,
#     explode=explode,
#     autopct="%.2f%%",
#     pctdistance=1.15,
#     labeldistance=1.35,
# )


# plt.figure(figsize=(6, 6))
# theta = np.linspace(0, 2.0 * np.pi, 1000)

# r = np.sin(6 * theta)
# plt.polar(theta, r)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

theta_max = 8 * np.pi
n = 1000
theta = np.linspace(0, theta_max, n)
x = theta
z = np.sin(theta)
y = np.cos(theta)
ax.plot(x, y, z, "g")

x = [5, 10, 15, 20]
z = [10, 0, 5, 15]
y = [0, 10, 5, 25]
s = [150, 130, 30, 160]
ax.scatter(x, y, z, s=s)

ax.scatter(x, y, s=s)

plt.show()

import seaborn as sns


data = sns.load_dataset("mpg")

sns.set_style("whitegrid")
plt.grid()
plt.show()


pd.DataFrame({'x': [1, 2, 3], 'y': [5, 6, 7]})
