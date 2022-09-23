import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import argsort, ndenumerate
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

def plot_correlation_image(subject, x, large=True):
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
    clb.set_label('Similarity', fontsize=titfs)
    plt.title(f'{subject}: {x.shape[0]} words', fontsize=titfs)


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

