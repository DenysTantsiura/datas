import pandas as pd

data = pd.Series([1, 2, 3])
data[3] = 4
print(data)
# --------------------------
data = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia'}
}

employees = pd.DataFrame(data)
employees['age'] = [25, 32, 19]
print(employees)
# --------------------------

data = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia'},
    'age': {'1': 25, '2': 32, '3': 19}
}

employees = pd.DataFrame(data)

new_employee = pd.Series(['Jhon', 'Denmark', 23], ['name', 'country', 'age'])

employees = employees.append(new_employee, ignore_index=True)
print(employees)
# --------------------------

numbers = pd.Series([1, 2, 3, 4, 5])
numbers.drop([1,3], inplace=True)
print(numbers)

# --------------------------
data = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia'},
    'age': {'1': 25, '2': 32, '3': 19}
}

employees = pd.DataFrame(data)

employees = employees.drop(['2'])
print(employees)

# --------------------------
data = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia'},
    'age': {'1': 25, '2': 32, '3': 19}
}

employees = pd.DataFrame(data)

employees = employees.drop(['age'], axis=1)
print(employees)
# --------------------------
data1 = {
    'name': {'1': 'Michael', '2': 'John'},
    'country': {'1': 'Canada', '2': 'USA'},
    'age': {'1': 25, '2': 32}
}

employees1 = pd.DataFrame(data1)

data2 = {
    'name': {'3': 'Liza', '4': 'Jhon'},
    'country': {'3': 'Australia', '4': 'Denmark'},
    'age': {'3': 19, '4': 23}
}

employees2 = pd.DataFrame(data2)

employees = pd.concat([employees1, employees2])

print(employees)
# --------------------------
data1 = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza', '4': 'Jhon'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia', '4': 'Denmark'}
}

employees1 = pd.DataFrame(data1)

data2 = {
    'age': {'1': 25, '2': 32, '3': 19, '4': 23}
}

employees2 = pd.DataFrame(data2)

employees = pd.concat([employees1, employees2], axis=1)

print(employees)
# --------------------------





# --------------------------

# --------------------------

