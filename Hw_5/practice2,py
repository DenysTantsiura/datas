from pathlib import Path
from threading import Thread
from typing import Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# %matplotlib inline


# def read_from_csvfile(file: Path) -> pd.DataFrame:
#     """Read content from csv-file and return dataframe."""
#     df = pd.read_csv('data/idle/idle-1.csv')

#     return df


# def get_data_to_dafaframe(path: Union[str, Path]) -> pd.DataFrame:
#     path = Path(path)
#     if not path.is_dir() or not path.exists():
#         return None
    
#     for file_system_object in path.iterdir():
#         if file_system_object.is_dir():
            
#             thread = Thread(target=get_data_to_dafaframe, args=(file_system_object, ))
#             thread.start()
            
#         elif file_system_object.suffix.lower() in ('.csv',):
#             thread_read_file = Thread(target=read_from_csvfile, args=(file_system_object, ))
#             thread_read_file.start()
        
#     return 1


# df = get_data_to_dafaframe('data')
# df or print('Sorry, no datasets found.')


def read_from_csvfile(file: Path) -> tuple:
    """Read content from csv-file and return file-name and dataframe in tuple."""
    df = pd.read_csv(file)

    return file.stem, df


def get_data_to_dafaframe(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.is_dir() or not path.exists():
        return None
    
    for file_system_object in path.iterdir():
        if file_system_object.is_dir():
            yield get_data_to_dafaframe(file_system_object)
            
        elif file_system_object.suffix.lower() in ('.csv',):
            yield read_from_csvfile(file_system_object)



dfs = [(data[0], data[1]) for data in next(get_data_to_dafaframe('data'))]

classification_human_activity = {}

def get_time_domain_features(y: str, df: pd.DataFrame) -> pd.DataFrame:
    if y not in classification_human_activity:
        classification_human_activity[y] = len(classification_human_activity)

    data = {
            'activity': [classification_human_activity[y]],
            'max_x': [df['accelerometer_X'].describe()['max']],
            'min_x': [df['accelerometer_X'].describe()['min']],
            'entropy_x': [-m.log(1/df['accelerometer_X'].shape[0])],
            'Interquartile_Range_x': [df['accelerometer_X'].describe()['25%']],
            'max_y': [df['accelerometer_Y'].describe()['max']],
            'min_y': [df['accelerometer_Y'].describe()['min']],
            'Mean_of_absolute_deviation_y': [df['accelerometer_Y'].describe()['mean']],
            'Median_y': [df['accelerometer_Y'].describe()['50%']],
            'Skewness_y': [1/df['accelerometer_Y'].describe()['std']],
            'Standard_deviation_y': [df['accelerometer_Y'].describe()['std']],
            'Root_mean_square_error_y': [abs(df['accelerometer_Y'].describe()['mean'])**0.5],
            'Skewness_z': [1/df['accelerometer_Z'].describe()['std']],
            }
    df_f = pd.DataFrame(data)
    return df_f


df = pd.DataFrame()
for el in dfs:
    df = pd.concat([df, get_time_domain_features(el[0].split('-')[0], el[1])], ignore_index=True)


