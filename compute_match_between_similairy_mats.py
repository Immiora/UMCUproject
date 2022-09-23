'''
compute similarity of similarity mats (across subjects and with edit distance)
'''

import numpy as np
import pandas as pd
import argparse
import os
from utils.plots import plot_corr_over_subjects
import matplotlib.pyplot as plt

def pd_get_upper(df):
    keep = np.triu(np.ones(df.shape), 1).astype('bool').reshape(df.size)
    return df.stack()[keep]

def main(subjects, n_syllables, cor_method):

    for n_syl in n_syllables:
        sim_all = dict.fromkeys(subjects)
        print(f'Number of syllables: {n_syl}')

        # load all similarity matrices
        for subject in subjects:
            sim_path = os.path.join('results', f'{subject.lower()}_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            assert os.path.exists(sim_path), 'file does not exist'
            sim_all[subject] = pd_get_upper(pd.read_csv(sim_path, sep=',', header=0, index_col=0))

        # add edit distance
        x, l = get_alt_feat_mat('edit_dist', n_syl, make_plot=True)
        assert len(l) == len(sim_all['F1']), 'edit distance is of difference length'
        sim_all['Edit_dist'] = pd.DataFrame(l, index=sim_all['F1'].index).stack().droplevel(2)

        # add phon edit distance
        x, l = get_alt_feat_mat('phon_edit_dist', n_syl, make_plot=True)
        assert len(l) == len(sim_all['F1']), 'edit distance is of difference length'
        sim_all['Phon_edit_dist'] = pd.DataFrame(l, index=sim_all['F1'].index).stack().droplevel(2)

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
    args = parser.parse_args()
    main(args.subjects, args.syllables, args.method)