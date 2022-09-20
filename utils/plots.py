import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import argsort


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

def plot_correlation_image(subject, x):
    order = argsort([i[::-1] for i in x.columns])
    plt.figure(figsize=(10, 10))
    im = plt.imshow(x.values[order][:, order], cmap='vlag')
    plt.yticks(range(len(x.columns)), x.columns[order])
    plt.xticks(range(len(x.index)), x.index[order], rotation=90)
    clb = plt.colorbar(im, fraction=0.046, pad=0.04)
    clb.set_label('Similarity', fontsize=12)
    plt.title(subject)