import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import argsort, ndenumerate, arange
import matplotlib.cm as cm
import seaborn as sns

def make_trajectory_plot(df_word_sensors, word):
    "df_word_sensors: df per word Nx12, N is time, 12 sensors"
    fig, ax = plt.subplots(1, 1)
    x = df_word_sensors
    for i_sen, s in enumerate(['UL', 'LL', 'JAW', 'TD', 'TB', 'TT']):
        ax.plot(x[s + 'x'], x[s + 'y'], label=s)

    plt.legend(['UL', 'LL', 'JAW', 'TD', 'TB', 'TT'], loc='lower left')
    ax.set_title(word)

def plot_image(x):
    plt.figure()
    plt.imshow(x, aspect='auto')

def plot_correlation_image(subject, x, large=True, label='Similarity'):
    order = argsort([i[::-1] for i in x.columns])
    figsize = (15, 15)
    fs = 10
    titfs = 16
    if large:
        figsize=(35, 35)
        fs = 5
        titfs = 20
    plt.figure(figsize=figsize)
    im = plt.imshow(x.values[order][:, order], cmap='vlag')
    plt.yticks(range(len(x.columns)), x.columns[order], fontsize=fs)
    plt.xticks(range(len(x.index)), x.index[order], rotation=90, fontsize=fs)
    clb = plt.colorbar(im, fraction=0.046, pad=0.04)
    clb.set_label(label, fontsize=titfs)
    plt.title(f'{subject}: {x.shape[0]} words', fontsize=titfs)

def plot_variance_image(subject, x, large=True, label='Variance'):
    plot_correlation_image(subject, x, large, label)


def plot_corr_over_subjects(x, show_values=False):
    plt.figure(figsize=(5, 5))
    im = plt.imshow(x.values, cmap='vlag', vmin=-1, vmax=1)
    plt.yticks(range(len(x.columns)), x.columns)
    plt.xticks(range(len(x.index)), x.index, rotation=90)
    if show_values:
        for (j, i), label in ndenumerate(x.values):
            plt.text(i, j, round(label, 2), ha='center', va='center')
            plt.text(i, j, round(label, 2), ha='center', va='center')
    clb = plt.colorbar(im, fraction=0.046, pad=0.04)
    clb.set_label('Similarity', fontsize=20)
    plt.title(f'all subjects')

def plot_pd_hist(subject, x):
    plt.figure()
    x.hist(bins=20)
    x.plot.kde(secondary_y=True, linewidth=2)
    plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.title(subject)

def plot_cluster_image(subject, x, clustering, large=True):
    order = argsort(clustering)
    figsize = (15, 15)
    fs = 10
    titfs = 16
    if large:
        figsize=(35, 35)
        fs = 5
        titfs = 20
    plt.figure(figsize=figsize)
    im = plt.imshow(x.values[order][:, order], cmap='vlag')
    plt.yticks(range(len(x.columns)), x.columns[order], fontsize=fs)
    plt.xticks(range(len(x.index)), x.index[order], rotation=90, fontsize=fs)
    clb = plt.colorbar(im, fraction=0.046, pad=0.04)
    clb.set_label('Similarity', fontsize=titfs)
    plt.title(f'{subject}: {x.shape[0]} words', fontsize=titfs)

    prev = 0
    for c in range(max(clustering)+1):
        rectangle = plt.Rectangle((prev-.5, prev-.5),
                                  sum(clustering==c), sum(clustering==c), fc='none', ec="red", lw=2)
        plt.gca().add_patch(rectangle)
        prev = prev+sum(clustering==c)


def plot_silhuette(clustering, silhouette_avg, sample_silhouette_values):
    plt.figure()
    y_lower = 10
    n_clusters = max(clustering) + 1
    for i in range(n_clusters)[::-1]:
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clustering == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()