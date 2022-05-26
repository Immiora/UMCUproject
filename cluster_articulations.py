'''
run basic clustering analysis
'''


import seaborn as sns
from utils.read_data import get_sensor_data, get_pos_data
from utils.plots import make_trajectory_plot, plot_image
from utils.preprocess_data import separate_by_syl, pad_data, compute_word_difference
from scipy.spatial import distance
from scipy.cluster import hierarchy

import Levenshtein



##
subject = 'M3'
subjects = ['F1', 'F5', 'M1', 'M3']

nsyllables = 5
normalize = True # global mean/std only


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


##
word_ind = list(syl_frames.keys())[0]
word = syl_frames[word_ind].word
make_trajectory_plot(frames[word_ind], word)
make_trajectory_plot(syl_frames[word_ind], word)


## pad each word's positions with global mean to have the same length
padded = pad_data(syl_frames)


## compute difference matix and correlations
correlations = compute_word_difference(padded)


## cluster
row_linkage = hierarchy.linkage(
    distance.pdist(correlations.values), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(correlations.values.T), method='average')

clustermap = sns.clustermap(correlations, row_linkage=row_linkage,
                            col_linkage=col_linkage,
                            dendrogram_ratio=(.1, .2),
                            cbar_pos=(.02, .32, .03, .2),
                            linewidths=0, figsize=(12, 12),
                            cmap='vlag')
clustermap.ax_row_dendrogram.remove()
clustermap.savefig(subject.lower() + '_cluster_global_mean.png', dpi=160)


## sort output correlation matrix
# order words according to the chronological order (key)

## plot and compare with edit distance
#



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