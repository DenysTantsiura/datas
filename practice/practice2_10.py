import matplotlib.pyplot as plt
import pandas as pd


date = pd.date_range(start='2021-09-01', freq='D', periods=8)
plt.plot(date, [23, 17, 17, 16, 15, 14, 17, 20], label='day temperature')

plt.xlabel('Дата', fontsize='small', color='midnightblue')
plt.ylabel('Температура', fontsize='small', color='midnightblue')
plt.title('Денна погода у м. Полтава', fontsize=15)
plt.text(date[0], 15, 'Осінь досить тепла', color="blue")
plt.legend()

plt.show()

# ---simple 2d + -------------
