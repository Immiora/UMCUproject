import os
import pandas as pd, numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from numpy import linalg as LA


def get_srate(file_number):
    directory = 'data/Data/F1/mat'

    # still needs to ignore the .DS_Store file in a better way
    file = sorted(os.listdir(directory))[file_number + 1]

    f = os.path.join(directory, file)
    mat = loadmat(f)['usctimit_ema_f1_{:03}_{:03}'.format(file_number * 5 + 1, file_number * 5 + 5)]

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


def make_trajectory_plot(word_dataframe, fixed_axes=False, twoD=False):
    x_UL, y_UL, z_UL = [], [], []
    for coordinate in word_dataframe['UL'][0]:
        x_UL.append(coordinate[0])
        y_UL.append(coordinate[1])
        z_UL.append(coordinate[2])

    x_LL, y_LL, z_LL = [], [], []
    for coordinate in word_dataframe['LL'][0]:
        x_LL.append(coordinate[0])
        y_LL.append(coordinate[1])
        z_LL.append(coordinate[2])

    x_JW, y_JW, z_JW = [], [], []
    for coordinate in word_dataframe['JW'][0]:
        x_JW.append(coordinate[0])
        y_JW.append(coordinate[1])
        z_JW.append(coordinate[2])

    x_TB, y_TB, z_TB = [], [], []
    for coordinate in word_dataframe['TB'][0]:
        x_TB.append(coordinate[0])
        y_TB.append(coordinate[1])
        z_TB.append(coordinate[2])

    x_TD, y_TD, z_TD = [], [], []
    for coordinate in word_dataframe['TD'][0]:
        x_TD.append(coordinate[0])
        y_TD.append(coordinate[1])
        z_TD.append(coordinate[2])

    x_TT, y_TT, z_TT = [], [], []
    for coordinate in word_dataframe['TT'][0]:
        x_TT.append(coordinate[0])
        y_TT.append(coordinate[1])
        z_TT.append(coordinate[2])

    # makes all the axis this size, currently
    if fixed_axes:
        ax.set_xlim3d(13, 15)
        ax.set_ylim3d(-72, -67)
        ax.set_zlim3d(-2.5, 2.5)

    if not twoD:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot3D(x_UL, y_UL, z_UL, label='UL')
        ax.plot3D(x_LL, y_LL, z_LL, label='LL')
        ax.plot3D(x_JW, y_JW, z_JW, label='JW')
        ax.plot3D(x_TB, y_TB, z_TB, label='TB')
        ax.plot3D(x_TD, y_TD, z_TD, label='TD')
        ax.plot3D(x_TT, y_TT, z_TT, label='TT')

    if twoD:
        plt.plot(x_UL, y_UL, label='UL')
        plt.plot(x_LL, y_LL, label='LL')
        plt.plot(x_JW, y_JW, label='JW')
        plt.plot(x_TB, y_TB, label='TB')
        plt.plot(x_TD, y_TD, label='TD')
        plt.plot(x_TT, y_TT, label='TT')

    plt.legend(loc='lower left')
    plt.title(str(word_dataframe['word'][0]) + ', ' + str(word_dataframe['sent'][0]))
    plt.show()


def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except KeyError:
        # if word not found in cmudict
        return syllables(word)


def syllables(word):
    # referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def longest(dictionary):
    biggest = 0
    for word in dictionary:
        length = len(dictionary[word].at[0, 'ULx'])
        if length > biggest:
            biggest = length

    return biggest


def longest_frame(dictionary):
    return max([i.shape[0] for i in dictionary.values()])


def frobenius_norm(matrix_a, matrix_b):
    difference_matrix = np.subtract(matrix_a, matrix_b)
    frob_norm = LA.norm(difference_matrix)
    return frob_norm