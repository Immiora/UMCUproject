'''
- load articulation data
- normalize data (optional)
- separate data into syllable groups
- plot 2d representations per word: greyscale, luminance represents time (%)
- downsample (nearest neighbors)
- plot all steps
'''

import seaborn as sns
from utils.read_data import get_sensor_data, get_pos_data
from utils.plots import make_trajectory_plot, plot_image
from utils.preprocess_data import separate_by_syl, pad_data, compute_word_difference
from scipy.spatial import distance
from scipy.cluster import hierarchy

import Levenshtein



##
subject = 'F1'
subjects = ['F1', 'F5', 'M1', 'M3']

nsyllables = 5
normalize = False # global mean/std only


## read data into a list of dataframes per articulator
# each item is of len = N of files,
# each df per file of shape = N of entries x positions (x,y,z)
UL_dfs, LL_dfs, JW_dfs, TD_dfs, TB_dfs, TT_dfs = get_sensor_data(subject)


## read in position data per word
# frames is a list of dataframes of len = N of words
# each df is of shape N timestamps x articulator and xy position (2 x 6 = 12)
frames = get_pos_data(subject, [UL_dfs, LL_dfs, JW_dfs, TD_dfs, TB_dfs, TT_dfs])


## get the list of all words
words=[frames[i].word for i in range(len(frames))]


## get subset of words with N of syllables == nsyllables
# dict with keys = indices of frames, values = selected frames
syl_frames = separate_by_syl(frames, nsyllables, normalize)


## discard color, visualize time as luminance
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import numpy as np

def map_time(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1) # 2nd dim shows shifts
    dydx = np.arange(1, x.shape[0])
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='Greys', norm=norm)
    lc.set_array(dydx)
    lc.set_linewidth(5)
    return lc

import pandas as pd
glbal_x_min = min(pd.concat(syl_frames).min().values[::2])
glbal_y_min = min(pd.concat(syl_frames).min().values[1::2])
glbal_x_max = max(pd.concat(syl_frames).max().values[::2])
glbal_y_max = max(pd.concat(syl_frames).max().values[1::2])

for key, val in syl_frames.items():

    x = val.values

    fig, ax = plt.subplots(1,1)
    ax.set_axis_off()
    lc = map_time(x[:, 0], x[:, 1])
    line = ax.add_collection(lc)
    #fig.colorbar(line, ax=ax)
    ax.set_xlim(glbal_x_min, glbal_x_max)
    ax.set_ylim(glbal_y_min, glbal_y_max)

    for s in range(2, 12, 2):
        newax = fig.add_axes(ax.get_position(), frameon=False)
        newax.set_axis_off()
        lc = map_time(x[:, s], x[:, s+1])
        line = newax.add_collection(lc)
        newax.set_xlim(glbal_x_min, glbal_x_max)
        newax.set_ylim(glbal_y_min, glbal_y_max)
    syl_frames[key].word

## save plot data into images



## downsample?



##
word_ind = list(syl_frames.keys())[0]
word = syl_frames[word_ind].word
make_trajectory_plot(frames[word_ind], word)
make_trajectory_plot(syl_frames[word_ind], word)


## compute difference matix and correlations
difmat_df = compute_word_difference(padded)
correlations = difmat_df.corr()
correlations_array = correlations.values


## cluster
row_linkage = hierarchy.linkage(
    distance.pdist(correlations_array), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(correlations_array.T), method='average')

clustermap = sns.clustermap(correlations, row_linkage=row_linkage,
                            col_linkage=col_linkage,
                            dendrogram_ratio=(.1, .2),
                            cbar_pos=(.02, .32, .03, .2),
                            linewidths=0, figsize=(12, 12),
                            cmap='vlag')
clustermap.ax_row_dendrogram.remove()



##
# clusters = hierarchy.fcluster(row_linkage, t=1)
# zipped = tuple(zip(clusters, labels))
#
# syl5_clusters = {}
#
# for pair in zipped:
#     if pair[0] in syl5_clusters:
#         syl5_clusters[pair[0]].append(pair[1])
#     else:
#         syl5_clusters[pair[0]] = [pair[1]]