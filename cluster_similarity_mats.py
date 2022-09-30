'''
cluster EMA using similarity mats
'''

import os
from utils.read_data import load_similarity_mat
from sklearn import cluster
import matplotlib.pyplot as plt
import argparse
from utils.pd_utils import pd_fix_columns
from utils.plots import plot_cluster_image, plot_silhuette
from sklearn.metrics import silhouette_samples, silhouette_score

def get_avg_silhuette(data, labels):
    return silhouette_score(data, labels)

def get_sample_silhuette(data, labels):
    return silhouette_samples(data, labels)

def main(n_syllables, methods):
    for method in methods:
        for n_syl in n_syllables[::-1]:
            sim_path = os.path.join('results', f'subjects_mean_similarity_mat_normalize_false_nsyl{str(n_syl)}.csv')
            data = pd_fix_columns(load_similarity_mat(sim_path))

            if method == 'affinity_propagation':
                cls = cluster.AffinityPropagation(affinity='precomputed')
                res = cls.fit(data)
                sil_avg = get_avg_silhuette(data, res.labels_)
                sil_smp = get_sample_silhuette(data, res.labels_)
                plot_silhuette(res.labels_, sil_avg, sil_smp)
                plot_cluster_image('cluster', data, res.labels_, large=True if n_syl < 4 else False)

            elif method == 'agglomerative':
                n_cluster_range = range(4, 20)
                for n_clusters in n_cluster_range:
                    cls = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
                    res = cls.fit(data)
                    sil_avg = get_avg_silhuette(data, res.labels_)
                    sil_smp = get_sample_silhuette(data, res.labels_)
                    plot_silhuette(res.labels_, sil_avg, sil_smp)
                    plot_cluster_image('cluster', data, res.labels_, large=True if n_syl < 4 else False)
            else:
                raise ValueError




##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--syllables', '-n', type=int,  nargs="+",
                        choices=range(1,6),
                        default=range(1,6),
                        help='Syllables to run')
    parser.add_argument('--methods', '-m', type=str,
                        choices=['affinity_propagation', 'agglomerative'],
                        default=['affinity_propagation', 'agglomerative'],
                        help='Clustering methods')
    args = parser.parse_args()
    main(args.syllables, args.methods)