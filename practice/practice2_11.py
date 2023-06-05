import matplotlib.pyplot as plt
import pandas as pd


date = pd.date_range(start='2021-09-01', freq='D', periods=8)
fig, axs = plt.subplots()
axs.plot(date, [23, 17, 17, 16, 15, 14, 17, 20], label='day temperature')
axs.plot(date, [19, 11, 16, 11, 10, 10, 11, 16], label='night temperature')
plt.xlabel('Дата', fontsize='small', color='midnightblue')
plt.ylabel('Температура', fontsize='small', color='midnightblue')
plt.title('Температура в м. Полтава', fontsize=15)
plt.text(date[0], 15, 'Осінь досить тепла', color="blue")
plt.legend()
plt.show()


date = pd.date_range(start='2021-09-01', freq='D', periods=8)
fig, axs = plt.subplots(2, 1)

axs[0].plot(date, [23, 17, 17, 16, 15, 14, 17, 20], label='day temperature')
axs[1].plot(date, [19, 11, 16, 11, 10, 10, 11, 16], label='night temperature')

axs[0].set_title('Денна', fontsize=10)
axs[1].set_title('Нічна', fontsize=10)

fig.suptitle('Температура в м. Полтава', fontsize=15)

plt.show()


# --simpe 2d double in 1 ------
