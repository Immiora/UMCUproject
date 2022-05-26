import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def make_trajectory_plot(df_word_sensors, word):
    "df_word_sensors: df per word Nx12, N is time, 12 sensors"
    fig, ax = plt.subplots(1, 1)
    x = df_word_sensors
    for i_sen, s in enumerate(['UL', 'LL', 'JW', 'TD', 'TB', 'TT']):
        ax.plot(x[s + 'x'], x[s + 'y'], label=s)

    plt.legend(['UL', 'LL', 'JW', 'TD', 'TB', 'TT'], loc='lower left')
    ax.set_title(word)

def plot_image(x):
    plt.figure()
    plt.imshow(x, aspect='auto')