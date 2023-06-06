import pandas as pd


contacts = pd.DataFrame(
    {
        'name': [
            'Allen Raymond',
            'Chaim Lewis',
            'Kennedy Lane',
            'Wylie Pope',
            'Cyrus Jackson',
        ],
        'email': [
            'nulla.ante@vestibul.co.uk',
            'dui.in@egetlacus.ca',
            'mattis.Cras@nonenimMauris.net',
            'est@utquamvel.net',
            'nibh@semsempererat.com',
        ],
        'phone': [
            '(992) 914-3792',
            '(294) 840-6685',
            '(542) 451-7038',
            '(692) 802-2949',
            '(501) 472-5218',
        ],
        'favorite': [False, False, True, False, True],
    },
    index=[1, 2, 3, 4, 5],
)


contacts.to_csv('data.csv', index=False)


users = pd.read_csv('data.csv')

print(users)

# --------------------------

contacts.to_excel('contacts.xlsx', sheet_name='Contacts')


persons = pd.read_excel('contacts.xlsx')

print(persons)

# --------------------------

data = {
    'name': {'1': 'Michael', '2': 'John', '3': 'Liza'},
    'country': {'1': 'Canada', '2': 'USA', '3': 'Australia'}
}

employees = pd.DataFrame(data)

employees.to_json('employees.json', orient='split')


employees = pd.read_json('employees.json', orient='split')

print(employees)

# --------------------------
