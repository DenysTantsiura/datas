import numpy as np


np.array([el for el in range(1, 11)])  # ndarray [1  2  3  4  5  6  7  8  9 10]
np.arange(1, 11)  # ndarray [1  2  3  4  5  6  7  8  9 10]
np.linspace(1, 10, num=10)  # ndarray [1.  2.  3.  4.  5  6.  7.  8.  9. 10.]

np.zeros((3, 5), dtype=int)  # ndarray [[0 0 0 0 0], [0 0 0 0 0], [0 0 0 0 0]]
np.random.randint(1, 11, size=(3, 5))  # ndarray [[6  4  4  2  3], [5  9  9  2 10], [10  4  8  2  8]]
np.random.rand(2, 2)  # ndarray [[0.16002009 0.35987033], [0.28608734 0.62922585]]

v2 = np.random.randint(1, 11, size=5)  # ndarray [6 1 6 5 5]
v3 = np.random.randint(1, 11, size=5)  # ndarray [1 2 8 3 7]
print('sum:\n', v2 + v3)  # ndarray [ 7  3 14  8 12]
print('subtraction:\n', v2 - v3)  # ndarray [ 5 -1 -2  2 -2]
print('multiplication:\n', v2 * v3)  # ndarray [ 6  2 48 15 35]
print(np.dot(v2, v3))  # 106 = 6+2+48+15+35
print(v2 @ v3)  # 106 

m6 = np.random.randint(1, 11, size=(3, 3))  # ndarray [[ 7  5  3],  [ 1 10  4],  [ 5  6  3]]
print(f'M6= \n{m6}\n')
m6_inv = np.linalg.inv(m6)  # ndarray [[ -1.2  -0.6   2. ], [ -3.4  -1.2   5. ], [  8.8   3.4 -13. ]]
print('inverse:')
print(m6_inv)

m6T = m6.T 

m10 = np.random.randint(1, 11, size=(3, 2))
m11 = np.random.randint(1, 11, size=(3, 2))
print(m10 * m11)
print(np.multiply(m10, m11))

print(m10.sum())
print(np.sum(m10))

print(np.array([[el.sum()] for el in m10[:]]))
print(np.array([[m10[el][:].sum()] for el in range(m10.shape[0])]))
print(np.sum(m10, axis=1).reshape((m10.shape[0], 1)))

print(m10 * m10)
print(m10 ** 2)
print(np.square(m10))

print(m10 ** 0.5)
print(np.sqrt(m10))

a = np.array([0, 2])
positive = a <= 1
print(positive)

print(a.shape[0])
# ------------------
"""
import pandas as pd

df = pd.read_csv('bikes_rent.csv')
df.head(5)

df['yr']  # Series
df[['yr']]  # DataFrame
df.loc[0]  # Series
df.iloc[1]  # Series
df.iloc[:, 1:4]  # DataFrame
df.iloc[0:5, 5:8]  # DataFrame
df.iloc[0:5]['yr']  # Series
df.iloc[:, [0, 1, 3]]  # DataFrame
df[0:3]  # DataFrame
df[df.index % 2 == 0]  # DataFrame
df[df['yr'] == 1]  # DataFrame
df[['yr', 'temp']]  # DataFrame
df[df.columns[:-1]]  # DataFrame
df.iloc[:, 1]  # Series
df.describe()
df.filter(items=[col for col in ('a', 'b', 'c'))])
"""
# df = pd.DataFrame({})
# df[df.columns[:-1]]