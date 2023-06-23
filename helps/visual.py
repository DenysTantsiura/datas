import random

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import seaborn as sns


# p1t1 = pd.read_html(
#                     io='https://uk.wikipedia.org/wiki/%D0%9D%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%BD%D1%8F_%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D0%B8',
#                     match='Коефіцієнт народжуваності в регіонах',
#                     )  #  poetry add lxml

# # print(p1t1[0].head())
# p1t1[0].replace(
#                 to_replace='—',
#                 value=np.nan,
#                 inplace=True,
#                 )
# p1t1z = p1t1[0].fillna(0)
# p1t1z.iloc[0:, 1:] = p1t1z.iloc[0:, 1:].astype('float64', errors='ignore')

# p1t1z.to_csv('helps/vdata.csv', index=False)

p1t1z = pd.read_csv('helps/vdata.csv')

def draw_3d(df: pd.DataFrame, x_column: str, y_label: str, z_label: str, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xl = np.array(df[x_column])
    yl = np.array(df.columns)[1:]
    plt.xticks(ticks=np.arange(0, xl.shape[0]), labels=xl, rotation=85)
    plt.yticks(ticks=np.arange(0, yl.shape[0]), labels=yl)
    plt.xlabel(x_column, fontsize='small', color='midnightblue')
    plt.ylabel(y_label, fontsize='small', color='midnightblue')
    ax.set_zlabel(z_label, fontsize='small', color='midnightblue')
    plt.title(title, fontsize=12)

    x, y = np.meshgrid(
        np.arange(0, xl.shape[0]), np.arange(1, yl.shape[0]))

    ax.plot_wireframe(
                    x,
                    y,
                    df.to_numpy()[x, y],
                    )

    plt.show()


draw_3d(p1t1z, 'Регіон', 'період', 'абс.народжуваність', title='Коефіцієнт народжуваності в регіонах України')


def draw_2d(df: pd.DataFrame, by_col: str, line: str, indicator: str, title: str) -> None:
    period = np.array(df.columns)[2:]
    plt.plot(period, np.array(
        np.array(df[df[by_col] == line].iloc[0][2:])), label=indicator)

    plt.xlabel('період (рік)', fontsize='small', color='midnightblue')
    plt.ylabel(indicator, fontsize='small', color='midnightblue')
    plt.title(title, fontsize=14)
    # plt.text(period[0], 15, '"неповні дані"', color="blue")
    plt.grid()
    plt.legend()

    plt.show()


draw_2d(p1t1z, 'Регіон', 'Київ', 'народжуваність', 'Народжуваність у м. Київ')


def draw_pie(df: pd.DataFrame, by_col: str, line: str, title: str) -> None:
    labels = np.array(df.columns)[1:]

    data = np.array(df[df[by_col] == line].iloc[0][1:])
    explode = [0 for _ in df[df[by_col] == line]][:-2] + [0.15]
    plt.pie(
        data,
        labels=labels,
        shadow=False,
        explode=explode,
        autopct="%.2f%%",
        pctdistance=1.10,
        labeldistance=1.25,
    )

    plt.title(title, fontsize=12)
    plt.show()

draw_pie(p1t1z, 'Регіон', 'Україна', title='Відносна народжуваність в Україні.')


def draw_bar(df: pd.DataFrame, by_col: str, type: str, indicator: str, title: str) -> None:
    # Побудуйте стовпчикову діаграму народжуваності по регіонах у 2019 році
    colors = []
    [colors.extend(['b', 'r', 'y', 'g'])
        for _ in range(len(df[by_col]) // 4 + 1)]
    plt.bar(
            df[by_col],
            df[type],
            color=colors[:len(df[by_col])]
            # angle=-0.5
            )

    plt.xlabel(by_col, fontsize='small', color='midnightblue')
    plt.ylabel(indicator, fontsize='small', color='midnightblue')
    plt.title(title, fontsize=12)
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


p1t1zm = p1t1z.replace(
                       to_replace=0,
                       value=np.nan,
                       inplace=False,
                       )
# Видаліть з таблиці дані по всій країні, останній рядок таблиці
new_p1t1 = p1t1zm.drop(p1t1zm.shape[0]-1)
# Замініть відсутні дані в стовпцях середніми значеннями цих стовпців (метод fillna)
new_p1t1 = new_p1t1.fillna({key: new_p1t1[key].mean() for key in new_p1t1.columns[1:]})


draw_bar(new_p1t1, 'Регіон', '2019', 'Коефіцієнт народжуваності', 'Коефіцієнт народжуваності в регіонах України')

# print(new_p1t1.head())
plt.hist(new_p1t1['2014'])
plt.xlabel('2014', fontsize='small', color='midnightblue')
plt.ylabel('quantity', fontsize='small', color='midnightblue')
plt.title('quantity of', fontsize=12)
plt.show()


p3t1 = pd.read_csv('helps/v2dataset.csv')
p3t1.columns = ['name', 'author', 'user_rating',
                'reviews', 'price', 'year', 'genre']
plt.figure()
sns.scatterplot(x='year', y='price', data=p3t1)
plt.show()

plt.figure()
sns.swarmplot(x='year', y='price', data=p3t1)
plt.show()

plt.figure()
sns.violinplot(x="year", y="price", data=p3t1)
plt.show()

plt.figure()
sns.set_style('darkgrid')
sns.scatterplot(x='reviews', y='price', data=p3t1)
plt.show()

"""
sns.pairplot(
             df,
             vars=['price', 'area', 'bedrooms', 'bathrooms', 'stories'],
             hue='basement'
             )
"""


def seaborn_displots(df: pd.DataFrame) -> None:
    for i in df:
        plt.figure(figsize=(6,6))
        sns.displot(df[i], kde=True, bins=int(df.shape[0]*0.045))
        plt.title(label=f'{i}[mean]= {df[i].mean()}')
        plt.show()
        print(df[i].describe())


def create_plots_in_one(data: np.array, title: str, xlabel: str, ylabel: str, figsize: tuple|None=None) -> None:
    """First column as x."""
    size = figsize or (15, 5)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 5)
    minor_ticks = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    x = list(map(lambda e: e[0], data))
    for el in range(1, data.shape[1]):
        y = list(map(lambda e: e[el], data))
        plt.plot(
                 x, 
                 y, 
                 label=f'mse-bloc-{el}', 
                 color=random.choice('rgbcmyk'), 
                 linestyle=random.choice(['-', '--', '-.', ':', None]),
                 marker=random.choice('o+xdvs*.^')
                 )

    plt.plot(x, np.mean(data[:, 1:], axis = 1), label='mean', color='k', linestyle=None)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    ax.grid(which='both')
    plt.show()


