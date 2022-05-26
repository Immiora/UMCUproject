
import math
import tqdm
import pandas as pd
import numpy as np
from scipy import io
from utils.general import *



def get_srate(subject, file_number):
    directory = os.path.join('data/Data', subject, 'mat')

    # made listdir_nohidden (in general) to take care of the problem with the hidden file
    file = sorted(listdir_nohidden(directory))[file_number]

    f = os.path.join(directory, file)
    mat = io.loadmat(f)[
        'usctimit_ema_' + subject.lower() + '_{:03}_{:03}'.format(file_number * 5 + 1, file_number * 5 + 5)]

    # returns the srate which is awkwardly stored here
    return mat[0][1][1][0][0]


def get_sensor_data(subject):
    directory = os.path.join('data/Data', subject, 'mat')
    counter = 1
    UL_df, LL_df, JW_df, TD_df, TB_df, TT_df = [], [], [], [], [], []

    for filename in sorted(listdir_nohidden(directory)):
        if filename.endswith('.mat'):
            f = os.path.join(directory, filename)
            mat = io.loadmat(f)
            # takes the data that is stored at the key that precedes the data for each .mat file
            data = mat['usctimit_ema_' + subject.lower() + '_{:03}_{:03}'.format(counter, counter + 4)]
            counter += 5

            # make dataframes of the six positions
            UL_df.append(pd.DataFrame.from_dict(data[0][1][2]))
            LL_df.append(pd.DataFrame.from_dict(data[0][2][2]))
            JW_df.append(pd.DataFrame.from_dict(data[0][3][2]))
            TD_df.append(pd.DataFrame.from_dict(data[0][4][2]))
            TB_df.append(pd.DataFrame.from_dict(data[0][5][2]))
            TT_df.append(pd.DataFrame.from_dict(data[0][6][2]))

    return UL_df, LL_df, JW_df, TD_df, TB_df, TT_df


def get_pos_data(subject, dataframes):

    def get_pos_list(position, dimension, file_number, starting_point, end_point):
        values = []
        if dimension == 'x':
            dim = 0
        elif dimension == 'y':
            dim = 1
        elif dimension == 'z':
            dim = 2
        else:
            raise ValueError

        index = positions.index(position)
        for i in range(end_point - starting_point):
            coordinate = (dataframes[index][file_number][dim][starting_point + i])
            if str(coordinate) != 'nan':
                values.append(coordinate)

        return np.array(values)

    frames = {}
    positions = ['UL', 'LL', 'JW', 'TB', 'TD', 'TT']
    sensors = ['ULx', 'ULy', 'LLx', 'LLy',
               'JWx', 'JWy', 'TDx', 'TDy',
               'TBx', 'TBy', 'TTx', 'TTy']

    # load timestamps per subject
    with open(os.path.join(subject + '_timestamps.txt'), 'r') as file:
        timestamps = file.read().splitlines()

        for word_number in tqdm.trange(len(timestamps)):
            split_line = timestamps[word_number].split(',')
            sent_number = int(split_line[-1])

            # find start and end by multiplying the timestamps with the sampling rate
            starting_point = math.floor(float(split_line[2]) * get_srate(subject, int(split_line[0])))
            end_point = math.ceil(float(split_line[3]) * get_srate(subject, int(split_line[0])))

            # make new dataframe for the current word
            df = pd.DataFrame()

            for sensor in sensors:
                # position, dimension, file_number, starting_point, end_point
                array = get_pos_list(sensor[:2], sensor[-1], int(split_line[0]), starting_point, end_point)
                df[sensor] = pd.Series(array)
                df.word = split_line[1]
                df.sent = int(split_line[-1])
                df.syl = get_nsyl(split_line[1])
                frames[word_number] = df

    return frames


