
import math
import warnings

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
    return mat[0][1][1][0][0] # only the first articulator

def get_sensor_data(subject):
    directory = os.path.join('data/Data', subject, 'mat')
    counter = 1
    sensors = {'UL':[], 'LL':[], 'JAW':[], 'TD':[], 'TB':[], 'TT':[]}

    for filename in sorted(listdir_nohidden(directory)):
        if filename.endswith('.mat'):
            f = os.path.join(directory, filename)
            mat = io.loadmat(f)
            # takes the data that is stored at the key that precedes the data for each .mat file
            data = mat['usctimit_ema_' + subject.lower() + '_{:03}_{:03}'.format(counter, counter + 4)]
            counter += 5

            # make dataframes of the six positions
            # data.dtype.descr
            # data[0][0]: audio, data[0][1-6]: sensors
            # data[0][i][0]: label of the stream (0 to 6 streams: sound + sensors): NAME
            # data[0][i][1]: sampling rate: SRATE
            # data[0][i][2]: data: SIGNAL
            for i in range(1, len(data[0])):
                sensors[data[0][i][0][0]].append(pd.DataFrame.from_dict(data[0][i][2]))

    return sensors

def get_pos_data(subject, dataframes, timestamps_file):

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

        for i in range(end_point - starting_point):
            if starting_point + i < len(dataframes[position][file_number][dim]):
                coordinate = (dataframes[position][file_number][dim][starting_point + i])
                if str(coordinate) != 'nan':
                    values.append(coordinate)
            else:
                warnings.warn(f'Requested end_point is greater than data length. File {file_number*5+1}-{file_number*5+5}')
                # check what is up here with F5: looks like audio file is longer than EMA data
                break

        return np.array(values)

    frames = {}
    positions = ['UL', 'LL', 'JAW', 'TD', 'TB', 'TT']
    sensors = ['ULx', 'ULy', 'LLx', 'LLy',
               'JAWx', 'JAWy', 'TDx', 'TDy',
               'TBx', 'TBy', 'TTx', 'TTy']

    # load timestamps per subject
    timestamps = pd.read_csv(os.path.join(subject + '_' + timestamps_file + '.txt'), sep=',', header=0)

    #for word_number in tqdm.trange(timestamps.shape[0]): #2784
    for word_number, row in tqdm.tqdm(timestamps.iterrows(), total=timestamps.shape[0]):
        #split_line = timestamps[word_number].split(',')
        #sent_number = row['sentence']#int(split_line[-1])

        # find start and end by multiplying the timestamps with the sampling rate
        # starting_point = math.floor(float(split_line[2]) * get_srate(subject, int(split_line[0])))
        # end_point = math.ceil(float(split_line[3]) * get_srate(subject, int(split_line[0]))) #max dur is 3601
        sr = get_srate(subject, int(row['file']))
        starting_point = math.floor(float(row['xmin']) * sr)
        end_point = math.ceil(float(row['xmax']) * sr) #max dur is 3601

        # make new dataframe for the current word
        df = pd.DataFrame()

        for sensor in sensors:
            # position, dimension, file_number, starting_point, end_point
            array = get_pos_list(sensor[:-1], sensor[-1], int(row['file']), starting_point, end_point)
            df[sensor] = pd.Series(array)
            df.file = row['file']
            df.word = row['text']
            df.sent = row['sentence']
            df.word_in_sent = row['word_in_sentence']
            df.syl = get_nsyl(row['text'])
        frames[word_number] = df

    return frames


