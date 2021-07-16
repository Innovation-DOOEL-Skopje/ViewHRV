from typing import List, Tuple
from collections import namedtuple
import numpy as np
import nolds
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.patches import Ellipse
import seaborn as sns

plt.ioff()

sns.set(font_scale=1, font="arial")

def plot_poincare(nn_intervals: List[float], plot_sd_features: bool = True):
    # For Lorentz / poincaré Plot
    ax1 = nn_intervals[:-1]
    ax2 = nn_intervals[1:]

    # compute features for ellipse's height, width and center
    dict_sd1_sd2 = get_poincare_plot_features(nn_intervals)
    sd1 = dict_sd1_sd2["sd1"]
    sd2 = dict_sd1_sd2["sd2"]
    mean_nni = np.mean(nn_intervals)

    # Plot options and settings
    style.use("seaborn-darkgrid")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.title("Poincaré / Lorentz Plot", fontsize=25)
    plt.xlabel('NN_n (s)', fontsize=20)
    plt.ylabel('NN_n+1 (s)', fontsize=20)
    plt.xlim(min(nn_intervals) - 10, max(nn_intervals) + 10)
    plt.ylim(min(nn_intervals) - 10, max(nn_intervals) + 10)

    # Poincaré Plot
    ax.scatter(ax1, ax2, c='b', s=2)

    if plot_sd_features:
        # Ellipse plot settings
        ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2 + 1,
                       height=2 * sd1 + 1, angle=45, linewidth=2,
                       fill=False)
        ax.add_patch(ells)

        ells = Ellipse(xy=(mean_nni, mean_nni), width=2 * sd2,
                       height=2 * sd1, angle=45)
        ells.set_alpha(0.05)
        ells.set_facecolor("blue")
        ax.add_patch(ells)

        # Arrow plot settings
        sd1_arrow = ax.arrow(mean_nni, mean_nni, -sd1 * np.sqrt(2) / 2, sd1 * np.sqrt(2) / 2,
                             linewidth=3, ec='r', fc="r", label="SD1")
        sd2_arrow = ax.arrow(mean_nni, mean_nni, sd2 * np.sqrt(2) / 2, sd2 * np.sqrt(2) / 2,
                             linewidth=3, ec='g', fc="g", label="SD2")

        plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")
    plt.savefig("static/Temporary/Poincare_Temporary.png")
    plt.close()
    return "Successfully Saved Poincare Plot"


def get_poincare_plot_features(nn_intervals: List[float]) -> dict:
    
    diff_nn_intervals = np.diff(nn_intervals)
    # measures the width of poincare cloud
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    # measures the length of the poincare cloud
    sd2 = np.sqrt(2 * np.std(nn_intervals, ddof=1) ** 2 - 0.5 * np.std(diff_nn_intervals, ddof=1) ** 2)
    ratio_sd2_sd1 = sd2 / sd1

    poincare_plot_features = {
        'sd1': sd1,
        'sd2': sd2,
        'ratio_sd2_sd1': ratio_sd2_sd1
    }

    return poincare_plot_features