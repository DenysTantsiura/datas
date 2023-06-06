from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Частина перша -> Hw2.1.ipynb
# Прочитайте дані за допомогою методу read_html з таблиці
# 'Коефіцієнт народжуваності в регіонах України (1950—2019)'
# ! таблиця за посиланням містить дані 1950-2014, 2014-2019 вишукувати самому, чи знехтувати за браком часу?
#  Якщо таблиці правильно оформлені, то метод дозволяє достатньо ефективно виконувати необхідну роботу.
# part 1 table 1 -> p1t1
p1t1 = pd.read_html(
    io='https://uk.wikipedia.org/wiki/%D0%9D%D0%B0%D1%80%D0%BE%D0%B4%D0%B6%D1%83%D0%B2%D0%B0%D0%BD%D1%96%D1%81%D1%82%D1%8C_%D0%B2_%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D1%96',
    match='Коефіцієнт народжуваності у регіонах',
)

# Вивести перші рядки таблиці за допомогою методу head
# значення абсолютні на 1000 осіб, а відображені на сторінці відсотки
print(p1t1[0].head())

# Визначте кількість рядків та стовпців у датафреймі (атрибут shape)
print(p1t1[0].shape)

# Замініть у таблиці значення '—' на значення NaN
# ? автоматично NaN стало, де були порожні комірки
print(p1t1[0].replace(
    to_replace=np.nan,
    value='-',
    # Whether to modify the DataFrame rather than creating a new one.
    inplace=False,
    limit=None,  # Maximum size gap to forward or backward fill
    regex=False,  #
    # method=_NoDefault.no_default
)
)
print(p1t1[0].replace(
    to_replace='-',
    value=np.nan,
)
)

# Визначте типи всіх стовпців за допомогою dataframe.dtypes
# float64 й int64 бо зчитувалось абсолютне значення, а відображались % ?
print(p1t1[0].dtypes)

# Замініть типи нечислових колонок на числові. Підказка - це колонки, де знаходився символ '—'
# ? Так замінили ж символ '—' на NaN. Комірок? Цілих колонок де всі значення NaN наче ж немає...
p1t1m1 = p1t1[0].fillna(0, inplace=False)
print(p1t1m1)

# xl = np.array(p1t1m1['регіон'])
# yl = np.array(p1t1m1.columns)[1:]
# np.arange(0, yl.shape[0])
p1t1m1.to_numpy() # [:, 1:]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

xl = np.array(p1t1m1['регіон'])
yl = np.array(p1t1m1.columns)[1:]
plt.xticks(ticks=np.arange(0, xl.shape[0]), labels=xl, rotation=85)
plt.yticks(ticks=np.arange(0, yl.shape[0]), labels=yl)
plt.xlabel('регіон', fontsize='small', color='midnightblue')
plt.ylabel('період', fontsize='small', color='midnightblue')
ax.set_zlabel('абс.народжуваність', fontsize='small', color='midnightblue')
plt.title('Коефіцієнт народжуваності в регіонах України (1950—2014)', fontsize=12)

x, y = np.meshgrid(
    np.arange(0, xl.shape[0]), np.arange(1, yl.shape[0]))

ax.plot_wireframe(
                  x,
                  y,
                  p1t1m1.to_numpy()[x, y],
                  )

# https://stackoverflow.com/questions/62185161/move-the-z-axis-on-the-other-side-on-a-3d-plot-python
# plt.show()

# print(pd.DataFrame(p1t1m1.to_numpy()[:, 1:]).replace(315, 0).max())
# p1t1m1.to_csv('test02-1.csv')


period = np.array(p1t1m1.columns)[1:]
plt.plot(period, np.array(
    p1t1m1[p1t1m1[0] == 'Київ'][1:]), label='народжуваність')

plt.xlabel('період (рік)', fontsize='small', color='midnightblue')
plt.ylabel('народжуваність', fontsize='small', color='midnightblue')
plt.title('Народжуваність у м. Київ', fontsize=14)
plt.text(period[0], 15, 'неповні дані', color="blue")
plt.legend()

plt.show()
