import os
import pandas as pd
from scipy.io import loadmat

def get_srate(file_number):
    directory = 'data/Data/F1/mat'

    # still needs to ignore the .DS_Store file in a better way
    file = sorted(os.listdir(directory))[file_number + 1]

    f = os.path.join(directory, file)
    mat = loadmat(f)['usctimit_ema_f1_{:03}_{:03}'.format(file_number *5 + 1, file_number *5 + 5)]

    # returns the srate which is stored here
    return mat[0][1][1][0][0]


def get_sensors():
    directory = 'data/Data/F1/mat'
    counter = 1
    UL_df, LL_df, JW_df, TD_df, TB_df, TT_df = [], [], [], [], [], []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.mat'):
            f = os.path.join(directory, filename)
            mat = loadmat(f)
            # takes the data that is stored at the key that precedes the data for each .mat file
            data = mat['usctimit_ema_f1_{:03}_{:03}'.format(counter, counter + 4)]
            counter += 5

            # make dataframes of the six positions
            UL_df.append(pd.DataFrame.from_dict(data[0][1][2]))
            LL_df.append(pd.DataFrame.from_dict(data[0][2][2]))
            JW_df.append(pd.DataFrame.from_dict(data[0][3][2]))
            TD_df.append(pd.DataFrame.from_dict(data[0][4][2]))
            TB_df.append(pd.DataFrame.from_dict(data[0][5][2]))
            TT_df.append(pd.DataFrame.from_dict(data[0][6][2]))

    return UL_df, LL_df, JW_df, TD_df, TB_df, TT_df


def get_key(val, dictionary):
    instances = []

    # retrieves the numbers of the instances of the word we are looking for in a list
    for key, value in dictionary.items():
        if val == value['word'][0]:
            instances.append(key)

    return instances