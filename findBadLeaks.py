import joblib
import os.path
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import numpy as np
import matplotlib.pyplot as plt


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df
leakdata = pd.read_feather('leak.feather')
leakdata = reduce_mem_usage(leakdata, use_float16=True)


if not os.path.exists('dictionary'):
    dictionary = dict()
    for i in range(1449):
        dictionary[i] = 0


    for i in range(len(leakdata)):
        if leakdata['meter_reading'][i] > dictionary.get(leakdata['building_id'][i]):
            dictionary[leakdata['building_id'][i]] = leakdata['meter_reading'][i]
        if i%1000 == 0:
            print('Row '+str(i) + ' of ' + str(len(leakdata)) + ' done')
    joblib.dump(dictionary, 'dictionary')
else:
    dictionary = joblib.load('dictionary')

plt.scatter(list(dictionary.keys()), list(dictionary.values()))
plt.xlabel('Building ID')
plt.ylabel('Meter Readings')
plt.show()
for i in range(1449):
    if dictionary.get(i)> 2000000:
        print('Building ID: ' + str(i) + ' Meter Reading: '+ str(dictionary.get(i)))

