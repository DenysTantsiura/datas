import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


pd.set_option('max_columns', 80)
# Частина третя: Аналіз датасет c Kaggle.com
# жирним шрифтом напис відповідь: потрібно буде вставити питання у файл і відповідь на нього.
# Наприклад:
# Яка бібліотека використовується для роботи з датафреймами у python? Відповідь: pandas

# Необхідно виконати:
# Пройти регістрацію на 
# https://www.kaggle.com/datasets/sootersaalu/amazon-top-50-bestselling-books-2009-2019
# завантажити датасет (csv файл)
# Прочитайте csv файл (використовуйте функцію read_csv)
p3t1 = pd.read_csv('dataset.csv')
print(p3t1)

# Виведіть перші п'ять рядків (використовується функція head)
print(p3t1.head())  # print(p3t1.head(5))

# Виведіть розміри датасету (використовуйте атрибут shape)
print(p3t1.shape)

# Відповідь: Про скільки книг зберігає дані датасет? Відповідь: p3t1.shape[0] -> 550
print(f'Відповідь: {p3t1.shape[0]}')  # 550

# Для кожної з книг доступні 7 змінних (колонок). Давайте розглянемо їх детальніше:
# Name - назва книги
# Author - автор
# User Rating - рейтинг (за 5-бальною шкалою)
# Reviews - кількість відгуків
# Price - ціна (у доларах станом на 2020 рік)
# Year - рік, коли книга потрапила до рейтингу Топ-50
# Genre - жанр
# Для спрощення подальшої роботи давайте трохи підправимо назви змінних. Як бачите, тут усі назви 
# починаються з великої літери, а одна - навіть містить пробіл. Це дуже небажано і може бути досить незручним.
#  Давайте змінимо регістр на малий, а пробіл замінимо на нижнє підкреслення (snake_style). А заразом і вивчимо 
# корисний атрибут датафрейму: 
# columns (можна просто присвоїти список нових імен цьому атрибуту)
p3t1.columns = ['name', 'author', 'user_rating', 'reviews', 'price', 'year', 'genre']
# p3t1 = p3t1.rename(columns = {'Name': 'name', 'User rating': 'user_rating'})

# Первинне дослідження даних
# Перевірте, чи у всіх рядків вистачає даних: виведіть кількість пропусків (na) у кожному зі стовпців (використовуйте функції isna та sum)
map_p3t1 = p3t1.isna()  # alias .isnull() 
print(map_p3t1)
print(p3t2 := np.sum(map_p3t1, axis=0))

# Відповідь: Чи є в якихось змінних пропуски? (Так / ні) Відповідь: ні
print(f'''Відповідь: {'Так' if p3t2.sum() > 0 else 'ні'}''')  # ні

# Перевірте, які є унікальні значення в колонці genre (використовуйте функцію unique)
#  Тобто унікальні з усіх, а не одиничні, що зустрічаються 1 раз серед усіх?
# https://pandas.pydata.org/docs/reference/api/pandas.unique.html
print(pd.unique(p3t1['genre']))

# Відповідь: Які є унікальні жанри? Відповідь: ['Non Fiction' 'Fiction']
unique_genre = np.insert(pd.unique(p3t1['genre']), 0, 'Відповідь:')
[print(el, end=' / ') for el in unique_genre]

# Тепер подивіться на розподіл цін: побудуйте діаграму (використовуйте kind='hist')
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# https://www.geeksforgeeks.org/matplotlib-pyplot-hist-in-python/
plt.hist(p3t1['price'])
plt.xlabel('price', fontsize='small', color='midnightblue')
plt.ylabel('quantity', fontsize='small', color='midnightblue')
plt.title('quantity of prices', fontsize=12)
plt.show()

# Визначте, яка ціна у нас максимальна, мінімальна, середня, медіанна (використовуйте функції max, min, mean, median)
print(
      np.max(p3t1['price']), '/',
      p3t1['price'].describe().loc['min'], '/',
      p3t1['price'].describe().loc['mean'], '/',
      p3t1['price'].describe().loc['50%'],
      )

# Відповідь: Максимальна ціна? Відповідь: 105
print(f'''Відповідь: {p3t1['price'].max()}''')  # 105.0

# Відповідь: Мінімальна ціна? Відповідь: 0
print(f'''Відповідь: {p3t1['price'].min()}''')  # 0

# Відповідь: Середня ціна? Відповідь: 13.1
print(f'''Відповідь: {p3t1['price'].mean()}''')  # 13.1

# Відповідь: Медіанна ціна? Відповідь: 11.0
print(f'''Відповідь: {p3t1['price'].median()}''')  # 11.0

# Пошук та сортування даних
# Відповідь: Який рейтинг у датасеті найвищий? Відповідь: 4.9
print(f'''Відповідь: {p3t1['user_rating'].max()}''')  # 4.9

# Відповідь: Скільки книг мають такий рейтинг? Відповідь: 52
print('Відповідь: ', p3t1['user_rating'][p3t1['user_rating'] == p3t1['user_rating'].max()].describe().loc['count'])  # 52

# Відповідь: Яка книга має найбільше відгуків? Відповідь:  # 534
print('Відповідь: ', p3t1[p3t1['reviews'] == p3t1['reviews'].max()])  # 534

# Відповідь: З тих книг, що потрапили до Топ-50 у 2015 році, яка книга найдорожча 
# (можна використати проміжний датафрейм)? Відповідь: 277
# https://datascientyst.com/get-top-10-highest-lowest-values-pandas/
group_1 = p3t1[p3t1['year'] == 2015]
group_1 = group_1.nlargest(n=50, columns=['user_rating'])
print('Відповідь: ', group_1[group_1['price'] == group_1['price'].max()])  # 277

# Відповідь: Скільки книг жанру Fiction потрапили до Топ-50 у 2010 році 
# (використовуйте &)? Відповідь: 20
print('Відповідь: ', p3t1[(p3t1['genre'] ==
      'Fiction') & (p3t1['year'] == 2010)].nlargest(n=50, columns=['user_rating']).shape[0])  # 20

# Відповідь: Скільки книг з рейтингом 4.9 потрапило до рейтингу у 2010 та 
# 2011 роках (використовуйте | або функцію isin)? Відповідь: 1
print('Відповідь: ', p3t1[(p3t1['year'] == 2010) | (p3t1['year'] == 2011)][p3t1['user_rating'] == 4.9].shape[0])  # 1
print('Відповідь: ', p3t1[p3t1['year'].isin([2010, 2011]) == True][p3t1['user_rating'] == 4.9].shape[0])  # 1

# І насамкінець, давайте відсортуємо за зростанням ціни всі книги, які потрапили до рейтингу 
# в 2015 році і коштують дешевше за 8 доларів (використовуйте функцію sort_values).
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
print(p3t3 := p3t1[(p3t1['year'] == 2015) & (p3t1['price'] < 8)].sort_values(by=['price']))

# Відповідь: Яка книга остання у відсортованому списку? Відповідь: 253
print(p3t3.tail(1))  # 253

# Агрегування даних та з'єднання таблиць...
# Остання секція цього домашнього завдання включає просунутіші функції. 
# Але не хвилюйтеся, pandas робить усі операції простими та зрозумілими.
# Для початку давайте подивимося на максимальну та мінімальну ціни для кожного з жанрів 
# (використовуйте функції groupby та agg, для підрахунку мінімальних та 
# максимальних значень використовуйте max та min). 
# Не беріть усі стовпці, виберіть тільки потрібні вам (???)
print(p3t1_gby := p3t1.groupby(['genre']))
print(p3t4 := p3t1_gby.agg({'price': ['min', 'max']}))

# Відповідь: Максимальна ціна для жанру Fiction: Відповідь 82
print('Відповідь: ', p3t4.loc['Fiction'][1])  # 82

# Відповідь: Мінімальна ціна для жанру Fiction: Відповідь 0
print('Відповідь: ', p3t4.loc['Fiction'][0])  # 0

# Відповідь: Максимальна ціна для жанру Non Fiction: Відповідь 105
print('Відповідь: ', p3t4.loc['Non Fiction'][1])  # 82

# Відповідь: Мінімальна ціна для жанру Non Fiction: Відповідь 0
print('Відповідь: ', p3t4.loc['Non Fiction'][0])  # 0

# Тепер створіть новий датафрейм, який вміщатиме 
# кількість книг для кожного з авторів (використовуйте функції groupby та agg, 
# для підрахунку кількості використовуйте count). 
# Не беріть усі стовпці, виберете тільки потрібні (???)
# https://sparkbyexamples.com/pandas/pandas-groupby-count-examples/
print(new_df := p3t1.groupby(['author'])['author'].count())

# Відповідь: Якої розмірності вийшла таблиця? Відповідь: 248
print('Відповідь: ', new_df.shape[0])  # 248

# Відповідь: Який автор має найбільше книг? Відповідь:  Jeff Kinney
print('Відповідь: ', new_df[new_df == new_df.max()].index[0])  # Jeff Kinney

# Відповідь: Скільки книг цього автора? Відповідь:  12
print('Відповідь: ', new_df.max())  # 12
print('Відповідь: ', new_df[new_df == new_df.max()][0])  # 12

# Тепер створіть другий датафрейм, який буде вміщати 
# середній рейтинг для кожного автора (використовуйте функції 
# groupby та agg, для підрахунку середнього значення використовуйте mean). 
# Не беріть усі стовпці, виберете тільки потрібні (???)
print(new_df2 := p3t1.groupby(['author'])['user_rating'].mean())

# Відповідь: У якого автора середній рейтинг мінімальний? Відповідь:
print('Відповідь: ', new_df2[new_df2 == new_df2.min()].index[0])  # Donna Tartt

# Відповідь: Який у цього автора середній рейтинг? Відповідь:
print('Відповідь: ', new_df2.min())  # 3.9
print('Відповідь: ', new_df2[new_df2 == new_df2.min()][0])  # 3.9

# З'єднайте останні два датафрейми так, щоб для кожного автора було видно 
# кількість книг та середній рейтинг (Використовуйте функцію concat з параметром axis=1). 
# Збережіть результат у змінну
p3t5 = pd.concat([new_df, new_df2], axis=1)
p3t5.columns = ['books_quantity', 'user_raiting']
print(p3t5)

# Відсортуйте датафрейм за зростаючою кількістю книг та зростаючим рейтингом 
# (використовуйте функцію sort_values)
print(p3t6 := p3t5.sort_values(by=['books_quantity', 'user_raiting']))

# Відповідь: Який автор перший у списку?  Відповідь: Muriel Barbery
print(p3t6.head(1).index[0])  # Muriel Barbery

# Робота здається у вигляді Jupyter файлу Hw2.3.ipynb
