# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd


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


