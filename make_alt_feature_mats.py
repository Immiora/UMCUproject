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
import textdistance
import tqdm
import re

def get_edit_dist(words):
    x = np.zeros((len(words), len(words)))
    for i1, w1 in enumerate(tqdm.tqdm(words)):
        for i2, w2 in enumerate(words):
            x[i1, i2] = -Levenshtein.distance(w1, w2)
    #l = x[np.triu(np.ones(x.shape), 1).astype('bool')]
    return x


def get_phon_edit_dist(words, stress=True):
    x = np.zeros((len(words), len(words)))
    for i1, w1 in enumerate(tqdm.tqdm(words)):
        for i2, w2 in enumerate(words):
            ph1 = check_word2phon(w1, lookup_phon(w1)[0])
            ph2 = check_word2phon(w2, lookup_phon(w2)[0])
            if stress == False:
                ph1 = [re.sub(r'\d+', '', i) for i in ph1]
                ph2 = [re.sub(r'\d+', '', i) for i in ph2]
            #x[i1, i2] = -compute_phoneme_edit_distance(ph1, ph2)
            x[i1, i2] = -textdistance.levenshtein.distance(ph1, ph2) # faster implementation (x10)
    return x


def main(metrics, n_syllables, make_plot=True):
    for metric in metrics:
        print(f'Metric: {metric}')

        for n_syl in n_syllables[::-1]:
            print(f'N syllables: {n_syl}')
            sim_path = os.path.join('results', f'f1_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            assert os.path.exists(sim_path), 'file does not exist'
            words = pd.read_csv(sim_path, sep=',', header=0, index_col=0).index
            if metric == 'edit_distance':
                x = get_edit_dist(words)
            elif metric == 'phoneme_edit_distance':
                x = get_phon_edit_dist(words, stress=True)
            elif metric == 'phoneme_edit_distance_nostress':
                x = get_phon_edit_dist(words, stress=False)
            else:
                raise ValueError

            if make_plot:
                plot_correlation_image(metric, pd.DataFrame(x, index=words, columns=words),
                                       large=True if n_syl < 4 else False)
                plt.savefig(f'pics/{metric}_similarity_mat_nsyl{str(n_syl)}_sorted.png', dpi=320)
                plt.close()

            pd.DataFrame(x, columns=words, index=words).to_csv(f'results/{metric}_similarity_mat_nsyl{str(n_syl)}.csv',
                                    sep=',', header=True, index=True)


##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--features', '-f', type=str,  nargs="+",
                        choices=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        default=['edit_distance', 'phoneme_edit_distance', 'phoneme_edit_distance_nostress'],
                        help='Subject to run')
    parser.add_argument('--syllables', '-n', type=int,  nargs="+",
                        choices=range(1,6),
                        default=range(1,6),
                        help='Syllables to run')
    args = parser.parse_args()
    main(args.features, args.syllables)