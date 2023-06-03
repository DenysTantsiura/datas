import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Частина друга: Аналіз файлів -> Hw2.2.ipynb
# Проведіть аналіз файлу 2017_jun_final.csv. Файл містить результати опитування розробників у червні 2017 року.
# Прочитайте файл 2017_jun_final.csv за допомогою методу read_csv
p2t2 = pd.read_csv('2017_jun_final.csv')
print(p2t2)

# Прочитайте отриману таблицю, використовуючи метод head
print(p2t2.head())

# Визначте розмір таблиці за допомогою методу shape
print(p2t2.shape)

# Визначте типи всіх стовпців за допомогою dataframe.dtypes
print(p2t2.dtypes)

# Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)
map_p2t2 = p2t2.isnull()  # alias .isna() 
print(map_p2t2)
print(p2t3 := np.sum(map_p2t2, axis=0))

# Видаліть усі стовпці з пропусками, крім стовпця 'Мова програмування'
print(new_p2t3 := p2t2.drop([idx for idx in p2t3.index if p2t3[idx] > 0 and idx != 'Язык.программирования'], axis=1, inplace=False))
    # print(new_p1t1 := p1t1[0].drop(p1t1[0].shape[0]-1))
    # print(p1t1[0].drop(['2014'], axis=1))

# Знову порахуйте, яка частка пропусків міститься в кожній колонці і переконайтеся, що залишився тільки стовпець 'Мова.програмування'
map_p2t3 = new_p2t3.isnull()  # alias .isna() 
print(map_p2t3)
print(p2t4 := np.sum(map_p2t3, axis=0))

# Видаліть усі рядки у вихідній таблиці за допомогою методу dropna
# лише що не мають всіх значень?:
p2t2 = p2t2.dropna(how='all')
print(p2t2)
# чи Взагалі всі?
p2t2 = p2t2.dropna()  # how='any' by default
print(p2t2)

# Визначте новий розмір таблиці за допомогою методу shape
print(p2t2.shape)

# Створіть нову таблицю python_data, в якій будуть тільки рядки зі спеціалістами, які вказали мову програмування Python
python_data = pd.read_csv('2017_jun_final.csv')
python_data.drop([idx for idx in python_data.index if python_data.iloc[idx]['Язык.программирования'] is np.nan], axis=0, inplace=True)
print(python_data)

# Визначте розмір таблиці python_data за допомогою методу shape
print(python_data.shape)

# Використовуючи метод groupby, виконайте групування за стовпчиком 'Посада'
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# print(python_data.groupby(by='Должность', axis=1))
# print(python_data.groupby(['Должность'], axis=1))
print(python_data_gby := python_data.groupby(['Должность']))

# Створіть новий DataFrame, де для згрупованих даних за стовпчиком 'Посада', виконайте агрегацію даних за допомогою методу agg і знайдіть мінімальне та максимальне значення у стовпчику 'Зарплата.в.місяць'
# https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html
print(python_data_gby.agg({'salary': ['min', 'max']}))

# Створіть функцію fill_avg_salary, яка повертатиме середнє значення заробітної плати на місяць. Використовуйте її для методу apply та створіть новий стовпчик 'avg'
# https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
# df.agg({'salary': ['mean']})
# https://stackoverflow.com/questions/60229375/solution-for-specificationerror-nested-renamer-is-not-supported-while-agg-alo
# Чи таки не просто середнє за місяць, а середнє за посадою? чи середнє за мовою?
def fill_avg_salary(df: Series) -> float | None:
    return df.describe().loc['mean']


python_data['avg'] = np.ones(python_data.shape[0]) * pd.DataFrame(python_data['salary']).apply(fill_avg_salary, axis=0).loc['salary']
print(python_data['avg'])

''' Ok 0
def fill_avg_salary(_, df: DataFrame) -> float:
    return df['salary'].agg(salary='mean')['salary']

# print(python_data.apply(fill_avg_salary, axis=1, args=(python_data,)))
python_data['avg'] = python_data.apply(fill_avg_salary, axis=1, args=(python_data,))  # [python_data.apply(fill_avg_salary, axis=1)) for _ in python_data.index]
print(python_data)
'''
def fill_avg_salary_by_position(df: DataFrame, el: str) -> float | None:
    df_c = df[df.loc[el]['Должность'] == df['Должность']]
    return df_c['salary'].agg(salary='mean')['salary']


python_data['avg_by_Должность'] = [fill_avg_salary_by_position(python_data, el) for el in python_data.index]
print(python_data['avg_by_Должность'])


# Створіть описову статистику за допомогою методу describe для нового стовпчика.
print(python_data.describe())

# Збережіть отриману таблицю в CSV файл
python_data.to_csv("data_p2.csv", index=False)
