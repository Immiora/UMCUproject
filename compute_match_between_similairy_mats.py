'''
compute similarity of similarity mats (across subjects and with edit distance)
'''

import pandas as pd
import argparse
import os
from utils.plots import plot_corr_over_subjects
from utils.pd_utils import pd_get_upper
from utils.read_data import load_similarity_mat
import matplotlib.pyplot as plt



def main(subjects, n_syllables, cor_method, features):

    for n_syl in n_syllables:
        sim_all = dict.fromkeys(subjects + features)
        print(f'Number of syllables: {n_syl}')

        # load all similarity matrices
        for subject in subjects:
            sim_path = os.path.join('results', f'{subject.lower()}_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            sim_all[subject] = pd_get_upper(load_similarity_mat(sim_path))

        # add edit distance metrics
        for feature in features:
            mat_path = os.path.join('results', f'{feature}_similarity_mat_nsyl{str(n_syl)}.csv')
            sim_all[feature] = pd_get_upper(load_similarity_mat(mat_path))

        out = pd.DataFrame(sim_all).corr(method=cor_method)
        print(out)
        plot_corr_over_subjects(out, show_values=True)
        plt.savefig(
            f'pics/all_subjects_similarity_mat_normalize_false_nsyl{str(n_syl)}_{cor_method}.png',
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
    parser.add_argument('--method', '-m', type=str,
                        choices=['pearson', 'kendall', 'spearman'],
                        default='pearson',
                        help='Correlation method')
    parser.add_argument('--features', '-f', type=str,  nargs="+",
                        choices=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        default=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        help='Subject to run')
    args = parser.parse_args()
    main(args.subjects, args.syllables, args.method, args.features)