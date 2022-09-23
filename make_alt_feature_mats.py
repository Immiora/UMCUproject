'''
compute similarity of similarity mats (across subjects and with edit distance)
'''

import numpy as np
import pandas as pd
import os
from utils.plots import plot_correlation_image
from utils.lang_utils import lookup_phon, check_word2phon, compute_phoneme_edit_distance
import matplotlib.pyplot as plt
import Levenshtein
import argparse


def get_edit_dist(words):
    x = np.zeros((len(words), len(words)))
    for i1, w1 in enumerate(words):
        for i2, w2 in enumerate(words):
            x[i1, i2] = -Levenshtein.distance(w1, w2)
    #l = x[np.triu(np.ones(x.shape), 1).astype('bool')]
    return x


def get_phon_edit_dist(words):
    x = np.zeros((len(words), len(words)))
    for i1, w1 in enumerate(words):
        for i2, w2 in enumerate(words):
            ph1 = lookup_phon(w1)[0]
            ph2 = lookup_phon(w2)[0]
            x[i1, i2] = -compute_phoneme_edit_distance(check_word2phon(w1, ph1), check_word2phon(w2, ph2))
    return x


def main(metrics, n_syllables, make_plot=True):
    for metric in metrics:
        print(f'Subject: {metric}')

        for n_syl in n_syllables:
            sim_path = os.path.join('results', f'f1_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            assert os.path.exists(sim_path), 'file does not exist'
            words = pd.read_csv(sim_path, sep=',', header=0, index_col=0).index
            if metric == 'edit_distance':
                x = get_edit_dist(words)
            elif metric == 'phoneme_edit_distance':
                x = get_phon_edit_dist(words)
            else:
                raise ValueError

            if make_plot:
                plot_correlation_image(metric, pd.DataFrame(x, index=words, columns=words),
                                       large=True if n_syl < 4 else False)
                plt.savefig(f'pics/phon_edit_dist_similarity_mat_normalize_nsyl{str(n_syl)}_sorted.png', dpi=320)
                plt.close()

            pd.DataFrame(x, columns=words, index=words).to_csv(f'results/{metric}_similarity_mat_nsyl{str(n_syl)}.csv',
                                    sep=',', header=True, index=True)


##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--features', '-f', type=str,  nargs="+",
                        choices=['edit_distance', 'phoneme_edit_distance'],
                        default=['edit_distance', 'phoneme_edit_distance'],
                        help='Subject to run')
    parser.add_argument('--syllables', '-n', type=int,  nargs="+",
                        choices=range(1,6),
                        default=range(1,6),
                        help='Syllables to run')
    args = parser.parse_args()
    main(args.features, args.syllables)