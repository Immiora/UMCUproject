'''
load similarity mats, plot distributions of values, plot mean and variance across mats,
find a way to merge mats across subjects into 1 for clustering?
'''

import pandas as pd
import argparse
import os
from utils.plots import plot_pd_hist, plot_variance_image, plot_correlation_image
from utils.read_data import load_similarity_mat
from utils.pd_utils import pd_fix_words, pd_get_upper
import matplotlib.pyplot as plt

def main(subjects, n_syllables, features):

    for n_syl in n_syllables[::-1]:
        sim_all = dict.fromkeys(subjects)
        feat_all = dict.fromkeys(features)
        print(f'Number of syllables: {n_syl}')

        # load all similarity matrices
        for subject in subjects:
            sim_path = os.path.join('results', f'{subject.lower()}_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            sim_all[subject] = load_similarity_mat(sim_path)

        # add edit distance metrics
        for feature in features:
            mat_path = os.path.join('results', f'{feature}_similarity_mat_nsyl{str(n_syl)}.csv')
            feat_all[feature] = load_similarity_mat(mat_path)

        # histogram of values per mat
        for subject in subjects:
            plot_pd_hist(subject, pd_get_upper(sim_all[subject]))
            plt.savefig(f'pics/{subject.lower()}_similarity_mat_normalize_false_nsyl{str(n_syl)}_hist.png', dpi=320)
            plt.close()

        for feature in features:
            plot_pd_hist(feature, pd_get_upper(feat_all[feature]))
            plt.savefig(f'pics/{feature}_similarity_mat_nsyl{str(n_syl)}_hist.png', dpi=320)
            plt.close()

        df_concat = pd.concat([sim_all[k].reset_index(drop=True) for k in sim_all.keys()])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = pd_fix_words(by_row_index.mean())
        df_var = pd_fix_words(by_row_index.var())

        # save mean
        df_means.to_csv(
            f'results/subjects_mean_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv',
            sep=',', header=True, index=True)

        # mean mats and variance
        plot_correlation_image('subjects mean', df_means, large=True if n_syl < 4 else False)
        plt.savefig(
            f'pics/all_subjects_similarity_mat_normalize_false_nsyl{str(n_syl)}_mean.png',
            dpi=320)
        plt.close()

        plot_variance_image('subjects var',  df_var, large=True if n_syl < 4 else False)
        plt.savefig(
            f'pics/all_subjects_similarity_mat_normalize_false_nsyl{str(n_syl)}_var.png',
            dpi=320)
        plt.close()


##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--subjects', '-s', type=str,  nargs="+",
                        choices=['F1', 'F5', 'M1', 'M3'],
                        default=['F1', 'F5', 'M1', 'M3'],
                        help='Subject to run')
    parser.add_argument('--syllables', '-n', type=int,  nargs="+",
                        choices=range(1,6),
                        default=range(1,6),
                        help='Syllables to run')
    parser.add_argument('--features', '-f', type=str,  nargs="+",
                        choices=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        default=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects, args.syllables, args.features)