# Домашнє завдання 1
import numpy as np


comment = '''\n1 Створіть одновимірний масив (вектор) 
		з першими 10-ма натуральними числами 
		та виведіть його значення.'''
print(comment)
# print(list(range(1, 11)))
# v1 = [el for el in range(1, 11)]  # [1, 2, 3, ...
# print(v1)
v1 = np.arange(1, 11)  # 1 2 3 ...
print('\nRESULT:')
print(v1)
v1 = np.linspace(1, 10, num=10)  # 1. 2. 3. ...
print('\nRESULT:')
print(v1)

comment = '''\n2 Створіть двовимірний масив (матрицю) розміром 3x3, 
		заповніть його нулями 
		та виведіть його значення.'''
print(comment)
m1 = np.zeros((3, 3), dtype=int)
print('\nRESULT:')
print(m1)

comment = '''\n3 Створіть масив розміром 5x5, 
		заповніть його випадковими цілими числами 
		в діапазоні від 1 до 10 
		та виведіть його значення.'''
print(comment)
# from random import randint
# m2 = [[randint(1, 10) for line in range(5)] for col in range(5)]
m2 = np.random.randint(1, 10, size=(5, 5))
print('\nRESULT:')
print(m2)

comment = '''\n4 Створіть масив розміром 4x4, 
		заповніть його випадковими дійсними числами 
		в діапазоні від 0 до 1 
		та виведіть його значення.'''
print(comment)
# from random import random
# m3 = [[random() for line in range(4)] for col in range(4)]
m3 = np.random.rand(4, 4)
print('\nRESULT:')
print(m3)

comment = '''\n5 Створіть два одновимірних масиви розміром 5, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та виконайте на них поелементні операції 
		додавання, віднімання та множення.'''
print(comment)
v2 = np.random.randint(1, 10, size=5)
v3 = np.random.randint(1, 10, size=5)
print(f'V2= \n{v2}\nV3= \n{v3}\n')
print('sum:\n', v2 + v3)
print('subtraction:\n', v2 - v3)
print('multiplication:\n', v2 * v3)

comment = '''\n6 Створіть два вектори розміром 7, 
		заповніть довільними числами 
		та знайдіть їх скалярний добуток.'''
print(comment)
v4 = 10 * np.random.rand(7)
v5 = np.array([el**0.5 for el in np.arange(7)])
print(f'V4= \n{v4}\nV5= \n{v5}\n')
print('\nRESULT:')
print(np.dot(v4, v5))

comment = '''\n7 Створіть дві матриці розміром 2x2 та 2x3, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та перемножте їх між собою.'''
print(comment)
m4 = np.random.randint(1, 10, size=(2, 2))
m5 = np.random.randint(1, 10, size=(2, 3))
print(f'M4= \n{m4}\nM5= \n{m5}\n')
print('\nRESULT:')
print(np.dot(m4, m5))

comment = '''\n8 Створіть матрицю розміром 3x3, 
		заповніть її випадковими цілими числами 
		в діапазоні від 1 до 10 
		та знайдіть її обернену матрицю.'''
print(comment)
m6 = np.random.randint(1, 10, size=(3, 3))
print(f'M6= \n{m6}\n')
m6_inv = np.linalg.inv(m6)
print('\nRESULT:')
print(m6_inv)

comment = '''\n9 Створіть матрицю розміром 4x4, 
		заповніть її випадковими дійсними числами 
		в діапазоні від 0 до 1 
		та транспонуйте її.'''
print(comment)
m7 = np.random.rand(4, 4)
print(f'M7= \n{m7}\n')
m7T = m7.T
print('\nRESULT:')
print(m7T)

comment = '''\n10 Створіть матрицю розміром 3x4 та вектор розміром 4, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та перемножте матрицю на вектор.'''
print(comment)
m8 = np.random.randint(1, 10, size=(3, 4))
v6 = np.random.randint(1, 10, size=4)
print(f'M8= \n{m8}\nV6= \n{v6}\n')
print('\nRESULT:')
print(np.dot(m8, v6))

comment = '''\n11 Створіть матрицю розміром 2x3 та вектор розміром 3, 
		заповніть їх випадковими дійсними числами 
		в діапазоні від 0 до 1 
		та перемножте матрицю на вектор.'''
print(comment)
m9 = np.random.rand(2, 3)
v7 = np.random.rand(3)
print(f'M9= \n{m9}\nV7= \n{v7}\n')
print('\nRESULT:')
print(np.dot(m9, v7))

comment = '''\n12 Створіть дві матриці розміром 2x2, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та виконайте їхнє поелементне множення.'''
print(comment)
m10 = np.random.randint(1, 10, size=(2, 2))
m11 = np.random.randint(1, 10, size=(2, 2))
print(f'M10= \n{m10}\nM11= \n{m11}\n')
print('\nRESULT:')
print(m10 * m11)

comment = '''\n13 Створіть дві матриці розміром 2x2, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та знайдіть їх добуток.'''
print(comment)
m12 = np.random.randint(1, 10, size=(2, 2))
m13 = np.random.randint(1, 10, size=(2, 2))
print(f'M12= \n{m12}\nM13= \n{m13}\n')
print('\nRESULT:')
print(np.dot(m12, m13))

comment = '''\n14 Створіть матрицю розміром 5x5, 
		заповніть її випадковими цілими числами 
		в діапазоні від 1 до 100 
		та знайдіть суму елементів матриці.'''
print(comment)
m14 = np.random.randint(1, 100, size=(5, 5))
print(m14)
print('\nRESULT:')
print(m14.sum())

comment = '''\n15 Створіть дві матриці розміром 4x4, 
		заповніть їх випадковими цілими числами 
		в діапазоні від 1 до 10 
		та знайдіть їхню різницю.'''
print(comment)
m15 = np.random.randint(1, 10, size=(4, 4))
m16 = np.random.randint(1, 10, size=(4, 4))
print(f'M15= \n{m15}\nM16= \n{m16}\n')
print('\nRESULT:')
print(m15 - m16)

comment = '''\n16 Створіть матрицю розміром 3x3, 
		заповніть її випадковими дійсними числами 
		в діапазоні від 0 до 1 
		та знайдіть вектор-стовпчик, що містить 
		суму елементів кожного рядка матриці.'''
print(comment)
m17 = np.random.rand(3, 3)
print(m17)
print('\nRESULT:')
print(np.array([el.sum() for el in m17[:]]))
print('\nRESULT:')
print(np.array([m17[el][:].sum() for el in range(m17.shape[0])]))

comment = '''\n17 Створіть матрицю розміром 3x4 
		з довільними цілими числами 
		і створінь матрицю з квадратами цих чисел.'''
print(comment)
m18 = np.random.randint(1, 10, size=(3, 4))
print(m18)
print('\nRESULT:')
print(m18 * m18)
print('\nRESULT:')
print(m18 ** 2)

comment = '''\n18 Створіть вектор розміром 4, 
		заповніть його випадковими цілими числами 
		в діапазоні від 1 до 50 
		та знайдіть вектор з квадратними коренями цих чисел.'''
print(comment)
v8 = np.random.randint(1, 50, size=4)
print(v8)
print('\nRESULT:')
print(v8 ** 0.5)

# Домашня робота здається у вигляді Jupyter файлу Hw1.ipynb.
