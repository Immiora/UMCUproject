'''
run basic clustering analysis
'''


import seaborn as sns
from utils.read_data import get_sensor_data, get_pos_data
from utils.plots import make_trajectory_plot, plot_image
from utils.preprocess_data import separate_by_syl, pad_data, compute_word_difference
from scipy.spatial import distance
from scipy.cluster import hierarchy
import pandas as pd
import numpy as np

#import Levenshtein

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

##
subject = 'F5'
subjects = ['F1', 'F5', 'M1', 'M3']

nsyllables = 5
normalize = False # global mean/std only
# normalization produces worse results? prioritize articulators with larger variance of movement (such as tongue) over
# lips, for example


## read data into a list of dataframes per articulator
# each item is of len = N of files,
# each df per file of shape = N of entries x positions (x,y,z)
UL_dfs, LL_dfs, JW_dfs, TD_dfs, TB_dfs, TT_dfs = get_sensor_data(subject)


## read in position data per word
# frames is a list of dataframes of len = N of words
# each df is of shape N timestamps x articulator and xy position (2 x 6 = 12)
frames = get_pos_data(subject, [UL_dfs, LL_dfs, JW_dfs, TD_dfs, TB_dfs, TT_dfs])

## test if the current subject has any words with missing data
missing = []

for frame in frames:
    if frames[frame].isnull().values.any():
        missing.append(frame)

# if len(missing) > 0:
#     raise Exception("Words found with missing data")
print(f'Number of all dataframes: {len(frames)}')
print(f'Number of dataframes with missing data: {len(missing)}')


## remove missing words
for mis in missing:
    frames.pop(mis, None)
print(f'Number of dataframes after removing dataframes with missing data: {len(frames)}')

## get the list of all words
words=[frames[i].word for i in frames.keys()]


## get subset of words with N of syllables == nsyllables
# dict with keys = indices of frames, values = selected frames
syl_frames = separate_by_syl(frames, nsyllables, normalize)


##
word_inds = [i for i in frames.keys() if frames[i].word == 'geological']
print(f'There are {len(word_inds)} entries of the specified word')
word_ind = word_inds[1]
word = syl_frames[word_ind].word
make_trajectory_plot(frames[word_ind], word)
make_trajectory_plot(syl_frames[word_ind], word)


## pad each word's positions with global mean to have the same length
padded = pad_data(syl_frames)


## compute difference matix and correlations
correlations = compute_word_difference(padded)


## plot
import matplotlib.pyplot as plt
order = np.argsort([i[::-1] for i in correlations.columns])
plt.figure(figsize=(10, 10))
im = plt.imshow(correlations.values[order][:, order], cmap='vlag')
plt.yticks(range(len(correlations.columns)), correlations.columns[order])
plt.xticks(range(len(correlations.index)), correlations.index[order], rotation=90)
clb = plt.colorbar(im, fraction=0.046, pad=0.04)
clb.set_label('Similarity', fontsize=12)
plt.title(subject)
plt.savefig(f'pics/{subject.lower()}_similarity_mat_normalize_{str(normalize).lower()}_sorted.png', dpi=160)

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
clustermap.savefig(f'pics/{subject.lower()}_cluster_normalize_{str(normalize).lower()}.png', dpi=160)


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