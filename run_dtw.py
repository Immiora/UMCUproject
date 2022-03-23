'''
Script to run dtw to align repetitions of the same word
Multi-dimensional representations are a challenge: Electrode x Coordinate x Time

using this package: https://github.com/pollen-robotics/dtw
'''


import math
import numpy as np
import matplotlib.pyplot as plt
from utils import * # can create a utils.py or a utils directory and imports utility functions from there (especially if you reuse them across notebooks/scripts)
import dtw
from scipy.spatial.distance import euclidean

def get_pos_list(position, dimension, file_number, starting_point, end_point):
    values = []
    if dimension == 'x':
        dim = 0
    elif dimension == 'y':
        dim = 1
    elif dimension == 'z':
        dim = 2  # raise ValueError for else
    else:
        raise ValueError

    if position == 'UL':
        for i in range(end_point - starting_point):
            coordinate = (UL_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    if position == 'LL':
        for i in range(end_point - starting_point):
            coordinate = (LL_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    if position == 'JW':
        for i in range(end_point - starting_point):
            coordinate = (JW_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    if position == 'TB':
        for i in range(end_point - starting_point):
            coordinate = (TB_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    if position == 'TD':
        for i in range(end_point - starting_point):
            coordinate = (TD_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    if position == 'TT':
        for i in range(end_point - starting_point):
            coordinate = (TT_df[file_number][dim][starting_point + i])
            values.append(coordinate)

    return values

## TODO: very slow currently, can improve?
UL_df, LL_df, JW_df, TD_df, TB_df, TT_df = get_sensors()

frames = {}
#word_number = 0

with open('timestamps.txt', 'r') as file:
    timestamps = file.read().splitlines()
    for word_number, line in enumerate(timestamps):  # check out enumerate()
        split_line = line.split(',')
        sent_number = int(split_line[-1])

        # find start and end by multiplying the timestamps with the sampling rate
        starting_point = math.floor(float(split_line[2]) * get_srate(int(split_line[0])))
        end_point = math.ceil(float(split_line[3]) * get_srate(int(split_line[0])))

        # make dataframe for each word, so 3481 dataframes
        # this is different from the one in main.ipynb, since it looks at x an y seperately, and
        # ignores the z values. still need to properly motivate this.
        data = {'word': [split_line[1]],
                'srate': [get_srate(int(split_line[0]))],
                'sent': [int(split_line[-1])],
                'ULx': [get_pos_list('UL', 'x', int(split_line[0]), starting_point, end_point)],
                'ULy': [get_pos_list('UL', 'y', int(split_line[0]), starting_point, end_point)],
                'LLx': [get_pos_list('LL', 'x', int(split_line[0]), starting_point, end_point)],
                'LLy': [get_pos_list('LL', 'y', int(split_line[0]), starting_point, end_point)],
                'JWx': [get_pos_list('JW', 'x', int(split_line[0]), starting_point, end_point)],
                'JWy': [get_pos_list('JW', 'y', int(split_line[0]), starting_point, end_point)],
                'TDx': [get_pos_list('TD', 'x', int(split_line[0]), starting_point, end_point)],
                'TDy': [get_pos_list('TD', 'y', int(split_line[0]), starting_point, end_point)],
                'TBx': [get_pos_list('TB', 'x', int(split_line[0]), starting_point, end_point)],
                'TBy': [get_pos_list('TB', 'y', int(split_line[0]), starting_point, end_point)],
                'TTx': [get_pos_list('TT', 'x', int(split_line[0]), starting_point, end_point)],
                'TTy': [get_pos_list('TT', 'y', int(split_line[0]), starting_point, end_point)]}

        df = pd.DataFrame(data)
        frames[word_number] = df
        #word_number += 1


## TODO: very slow currently, can improve?
word_instances = {}

for frame in frames:
    # continues if the instances of this word have already been found
    if frames[frame]['word'][0] in word_instances:
        continue

    # saves all the key numbers of instances of the words
    word_instances[frames[frame]['word'][0]] = get_key(frames[frame]['word'][0], frames)


##
# word_instances['often']
# [207, 1140, 1144, 2173, 2873, 2975, 3080, 3256]

# tested for 2873 and 1140 of often
rep1 = 0
rep2 = 1184
x1, x2 = [], []
for sensor in ['UL', 'LL', 'JW', 'TD', 'TB', 'TT']:
    x1.append(np.stack([frames[rep1][sensor+'x'][0],frames[rep1][sensor+'y'][0]]))
    x2.append(np.stack([frames[rep2][sensor+'x'][0],frames[rep2][sensor+'y'][0]]))
x1 = np.concatenate(x1).T # 0:ULx, 1:ULy, 2:LLx, 3:LLy etc
x2 = np.concatenate(x2).T

dist, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(x1, x2, dist=euclidean)
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()

print(dist)
x1_new = x1[path[0]]
x2_new = x2[path[1]]

print(x1.shape)
print(x2.shape)
print(x1_new[:,0].shape)
print(x2_new[:,0].shape)

##
plt.figure()
plt.subplot(121)
plt.plot(x1[:, 0], label='original ULx')
plt.plot(x1_new[:, 0], label='dtw ULx')
plt.legend()
plt.title('repetition 1')

plt.subplot(122)
plt.plot(x1[:, 1], label='original ULy')
plt.plot(x1_new[:, 1], label='dtw ULy')
plt.legend()
plt.title('repetition 1')

##
plt.figure()
plt.subplot(121)
plt.plot(x1[:, 0], x1[:, 1], label='repetition 1')
plt.plot(x2[:, 0], x2[:, 1], label='repetition 2')
plt.legend()
plt.title('original UL')

plt.subplot(122)
plt.plot(x1_new[:,0], x1_new[:,1], label='repetition 1')
plt.plot(x2_new[:,0], x2_new[:,1], label='repetition 2')
plt.legend()
plt.title('dtw UL')

##
x1_d3 = x1.T.reshape(6, 2, -1)
x2_d3 = x2.T.reshape(6, 2, -1)
x1_new_d3 = x1_new.T.reshape(6, 2, -1)
x2_new_d3 = x2_new.T.reshape(6, 2, -1)

##
for i_rep, (a, b) in enumerate(zip([x1_d3, x2_d3], [x1_new_d3, x2_new_d3])):
    fig, ax = plt.subplots(1, 2)
    for i_sen, sensor in enumerate(['UL', 'LL', 'JW', 'TD', 'TB', 'TT']):
        ax[0].plot(a[i_sen, 0], a[i_sen, 1], label=sensor)
        ax[1].plot(b[i_sen, 0], b[i_sen, 1], label=sensor)
    plt.legend(['UL', 'LL', 'JW', 'TD', 'TB', 'TT'], loc='lower left')
    ax[0].set_title('repetition ' + str(i_rep+1) + ' original')
    ax[1].set_title('repetition ' + str(i_rep+1) + ' dtw')

##
# plt.figure()
# plt.plot(np.array(x1)[1,1])
#
# plt.figure()
# plt.plot(x1_c[:, 3])