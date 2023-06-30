import math as m
from pathlib import Path
import pickle
from typing import Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
# %matplotlib inline


def read_from_csvfile(file: Path) -> pd.DataFrame:
    """Read content from csv-file and return dataframe from content."""
    df = pd.read_csv(file)
    
    return df



df = pd.DataFrame()
classification_human_activity = {}


def get_statistical_features(y: str, df: pd.DataFrame) -> pd.DataFrame:
    # y - type real data; df - sample data. 
    if y not in classification_human_activity:
        classification_human_activity[y] = len(classification_human_activity)

    data = {
            'activity': [classification_human_activity[y]],
            # 'max_x': [max(df['accelerometer_X'])],  # [df['accelerometer_X'].describe().loc['max']],
            # 'min_x': [min(df['accelerometer_X'])],  # [df['accelerometer_X'].describe().loc['min']],
            # 'mean_x': [df['accelerometer_X'].mean(axis=0)],  # [df['accelerometer_X'].describe().loc['mean']],
            # 'med_x': [df['accelerometer_X'].median(axis=0)],  # median
            # 'std_x': [df['accelerometer_X'].std(axis=0)],  # Standard deviation !!= [df['accelerometer_Y'].describe().loc['std']], 
            # 'skew_x': [df['accelerometer_X'].skew(axis=0)],  # skewness
            # 'kurt_x': [df['accelerometer_X'].skew(axis=0)],  # kurtosis
            # 'var_x': [df['accelerometer_X'].var(axis=0)],  # variance
            # 'idxmax_x': [df['accelerometer_X'].idxmax(axis=0)],  # index of first occurrence of maximum over requested axis
            # 'idxmin_x': [df['accelerometer_X'].idxmin(axis=0)],  # index of first occurrence of minimum over requested axis
            }
    [data.update({
                  f'max_{col[-1]}':[max(df[col])],
                  f'min_{col[-1]}':[min(df[col])],
                  f'mean_{col[-1]}':[df[col].mean(axis=0)],
                  f'med_{col[-1]}':[df[col].median(axis=0)],  # median
                  f'std_{col[-1]}':[df[col].std(axis=0)],  # Standard deviation
                  f'skew_{col[-1]}':[df[col].skew(axis=0)],  # skewness
                  f'kurt_{col[-1]}':[df[col].kurt(axis=0)],  # kurtosis
                  f'var_{col[-1]}':[df[col].var(axis=0)],  # variance
                  f'idxmax_{col[-1]}':[df[col].idxmax(axis=0)],  # index of first occurrence of maximum over requested axis
                  f'idxmin_{col[-1]}':[df[col].idxmin(axis=0)],  # index of first occurrence of minimum over requested axis
                #   f'rmse_{col[-1]}':[mean_squared_error(df[col], np.array([df[col].mean(axis=0) for _ in range(df.shape[0])]), squared=False)],  # Root Mean Square Error
                #   f'mae_{col[-1]}':[mean_absolute_error(df[col], np.array([df[col].mean(axis=0) for _ in range(df.shape[0])]))],  # mean absolute error
                  }) 
        for col in df.columns]

    [data.update({
                  f'rmse_{col[-1]}':[mean_squared_error(df[col], [data[f'mean_{col[-1]}'] for __ in range(df.shape[0])], squared=False)],  # variance# Root Mean Square Error
                  f'mae_{col[-1]}':[mean_absolute_error(df[col], [data[f'mean_{col[-1]}'] for __ in range(df.shape[0])])],  # mean absolute error
                  }) 
        for col in df.columns]
    df_f = pd.DataFrame(data)
    
    return df_f


def get_data_to_dafaframe(path: Union[str, Path], df: pd.DataFrame) -> pd.DataFrame:
    path = Path(path)
    if not path.is_dir() or not path.exists():
        return None
    
    for file_system_object in path.iterdir():
        if file_system_object.is_dir():
            df = get_data_to_dafaframe(file_system_object, df)
            
        elif file_system_object.suffix.lower() in ('.csv',):
            df_add = read_from_csvfile(file_system_object)
            df = pd.concat([df, get_statistical_features(file_system_object.stem.split('-')[0], df_add)], ignore_index=True)

    return df



def save_prepared_data(df: pd.DataFrame, file_name: str='data1.bin') -> None:
    with open(file_name, 'wb') as fh:
        pickle.dump(df, fh)



def load_prepared_data(file_name: str='data1.bin') -> pd.DataFrame:
    with open(file_name, 'rb') as fh:
        df = pickle.load(fh)

    return df



if Path('data0.bin').is_file() and Path('data1.bin').is_file():
    df = load_prepared_data('Hw_5/data1.bin')
    classification_human_activity = load_prepared_data('Hw_5/data0.bin')

else:
    df = get_data_to_dafaframe('data', df)
    save_prepared_data(df)
    save_prepared_data(classification_human_activity, 'data0.bin')


if not classification_human_activity.get(0):
    classification_human_activity = {val:key for key, val in classification_human_activity.items()}

classification_human_activity


X = df.iloc[:, 1:]
y = df.iloc[:, 0]
#
y = y.values

# Робимо вибірки - розділяємо всі дані на групи для тренування, валідаційну та тестову
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)  # random_state=42
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape

model = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
y_valid_pred = model.predict_proba(X_valid)  # [:, 2]  # probability for classification_human_activity[2] - для 'idle'
print('-type(y_valid_pred)- '*5)
print(type(y_valid_pred))
print('- - '*32)
print(y_valid_pred.head() if isinstance(y_valid, pd.DataFrame) else y_valid)
print('- - '*32)
a = pd.DataFrame(y_valid_pred)
print('-a.head()- '*10)
print(a.head())
print('-a.idxmax(1)- '*10)
print(a.idxmax(1))
print('- - '*32)
print(a.idxmax(1).values)
print('- - '*32)

# from_model_by_max = pd.DataFrame(y_valid_pred).idxmax(1).values

#....
# Тренуємо моделі (ймовірність - probability)
# C = 100.  # коефіцієнт ваги функції втрат (як вірогіднісний), <1 менша точність, щоб алгоритм більше орієнтувався на margin ; більший - більша точність, але!
# svc_linear = SVC(kernel='linear', C=C, probability=True).fit(X_train, y_train)  # kernel - ядрова функція для викривлення простору даних, щоб побудувати простішу гіперплощину - простішу лінійну модель
# svc_rbf = SVC(kernel='rbf', C=C, gamma=0.7, probability=True).fit(X_train, y_train)
# svc_poly = SVC(kernel='poly', degree=3, C=C, probability=True).fit(X_train, y_train)
# # for clf in (svc_linear, svc_rbf, svc_poly):
# #     h = clf.predict()


# ..................

# svc_linear = SVC(kernel='linear', C=100, probability=True).fit(X_train, y_train)
# y_valid_pred = svc_linear.predict_proba(X_valid)  # y_valid_pred = svc_linear.predict(X_valid)[:, 1]

# roc_auc_score(y_valid, pd.DataFrame(y_valid_pred).idxmax(1).values)  # y_valid_pred

# roc_auc_score(real_result, y_valid_pred)  #! multi_class must be in ('ovo', 'ovr')
