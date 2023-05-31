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

# print(type(p1t1))  # <class 'list'>
# print(type(p1t1[0]))  # <class 'pandas.core.frame.DataFrame'>

# Вивести перші рядки таблиці за допомогою методу head
print(p1t1[0].head())  # чомусь всі значення в 10 разів більші !

# Визначте кількість рядків та стовпців у датафреймі (атрибут shape)
print(p1t1[0].shape)

# Замініть у таблиці значення '—' на значення NaN
# ? автоматично NaN стало, де були порожні комірки
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
print(p1t1[0].replace(
                      to_replace=np.nan, 
                      value='-', 
                      inplace=False, # Whether to modify the DataFrame rather than creating a new one.
                      limit=None, # Maximum size gap to forward or backward fill
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
print(p1t1[0].dtypes)  # float64 й int64 бо при зчитуванні всі значення в 10 разів більші

# Замініть типи нечислових колонок на числові. Підказка - це колонки, де знаходився символ '—'
# ? Так замінили ж символ '—' на NaN. Комірок? Цілих колонок де всі значення NaN наче ж немає...
p1t1m1 = p1t1[0].fillna(0, inplace=False)
print(p1t1m1)

# Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html?highlight=isnull#pandas.DataFrame.isnull
map_p1t1 = p1t1[0].isna()  # alias .isnull()
print(map_p1t1)
print(np.sum(map_p1t1, axis=0))

# Видаліть з таблиці дані по всій країні, останній рядок таблиці
##- print(p1t1[0].drop(['Україна'], axis=0))  # p1t1[0].drop(['Україна'])
# print(p1t1[0].drop([27], axis=0))
print(new_p1t1 := p1t1[0].drop(p1t1[0].shape[0]-1))
# print(p1t1[0].drop(['2014'], axis=1))

# Замініть відсутні дані в стовпцях середніми значеннями цих стовпців (метод fillna)
'''
new_p1t1 = new_p1t1.fillna({'1950': new_p1t1['1950'].mean(), 
                            '1960': new_p1t1['1960'].mean(),
                            ... 
                            })  # new_p1t1['регіон'][1:]
'''
new_p1t1 = new_p1t1.fillna({key: new_p1t1[key].mean() for key in new_p1t1.columns[1:]})
print(new_p1t1)

# Отримайте список регіонів, де рівень народжуваності у 2019 році був вищим за середній по Україні
# в таблиці дані лише до 2014 включно...
print(new_p1t1[new_p1t1['2014'] > new_p1t1['2014'].mean()]['регіон'])

# У якому регіоні була найвища народжуваність у 2014 році?
print(new_p1t1[new_p1t1['2014'] == new_p1t1['2014'].max()]['регіон'])

# Побудуйте стовпчикову діаграму народжуваності по регіонах у 2019 році
# в таблиці дані лише до 2014 включно...
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
# https://www.delftstack.com/ru/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/
# https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
plt.bar(
        new_p1t1['регіон'],
        new_p1t1['2014'],
        # color=['b', 'r', 'y', 'g'],
        # angle=-0.5
        )

plt.xlabel('регіон', fontsize='small', color='midnightblue')
plt.ylabel('Коефіцієнт народжуваності', fontsize='small', color='midnightblue')
plt.title('Коефіцієнт народжуваності в регіонах України (1950—2014)', fontsize=12)
plt.xticks(rotation=85)
plt.grid(True)
plt.subplots_adjust(
                    top=0.935,
                    bottom=0.49,
                    left=0.14,
                    right=0.81,
                    hspace=0.2,
                    wspace=0.2
                    )
plt.show()

# Частина друга: Аналіз файлів -> Hw2.2.ipynb
# Проведіть аналіз файлу 2017_jun_final.csv. Файл містить результати опитування розробників у червні 2017 року.
# Прочитайте файл 2017_jun_final.csv за допомогою методу read_csv
#!!! to randome colors in last plt-bar!!!
q = pd.read_csv('2017_jun_final.csv')
print(q)

# Прочитайте отриману таблицю, використовуючи метод head


# Визначте розмір таблиці за допомогою методу shape


# Визначте типи всіх стовпців за допомогою dataframe.dtypes


# Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)


# Видаліть усі стовпці з пропусками, крім стовпця 'Мова програмування'


# Знову порахуйте, яка частка пропусків міститься в кожній колонці і переконайтеся, що залишився тільки стовпець 'Мова.програмування'


# Видаліть усі рядки у вихідній таблиці за допомогою методу dropna


# Визначте новий розмір таблиці за допомогою методу shape


# Створіть нову таблицю python_data, в якій будуть тільки рядки зі спеціалістами, які вказали мову програмування Python


# Визначте розмір таблиці python_data за допомогою методу shape


# Використовуючи метод groupby, виконайте групування за стовпчиком 'Посада'


# Створіть новий DataFrame, де для згрупованих даних за стовпчиком 'Посада', виконайте агрегацію даних за допомогою методу agg і знайдіть мінімальне та максимальне значення у стовпчику 'Зарплата.в.місяць'


# Створіть функцію fill_avg_salary, яка повертатиме середнє значення заробітної плати на місяць. Використовуйте її для методу apply та створіть новий стовпчик 'avg'


# Створіть описову статистику за допомогою методу describe для нового стовпчика.


# Збережіть отриману таблицю в CSV файл


