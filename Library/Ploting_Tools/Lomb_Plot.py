from collections import namedtuple
from typing import List
from typing import Tuple
from astropy.timeseries import LombScargle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from Library.utils import _get_freq_psd_from_nn_intervals
from Library.utils import _create_timestamp_list

sns.set(font_scale=1, font="arial")

# Named Tuple for different frequency bands
UlfBand = namedtuple("Ulf_band", ["low", "high"])
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])

# Frequency Methods name
WELCH_METHOD = "welch"
LOMB_METHOD = "lomb"

def Psd_Plot(nn_intervals: List[float], method: str = "lomb", sampling_frequency: int = 7,
             interpolation_method: str = "linear", ulf_band: namedtuple = UlfBand(0, 0.00333), vlf_band: namedtuple = VlfBand(0.00333, 0.04),
             lf_band: namedtuple = LfBand(0.04, 0.15), hf_band: namedtuple = HfBand(0.15, 0.40)):

    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method)

    # Calcul of indices between desired frequency bands
    ulf_indexes = np.logical_and(freq >= ulf_band[0], freq < ulf_band[1])
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    frequency_band_index = [ulf_indexes, vlf_indexes, lf_indexes, hf_indexes]
    label_list = ["ULF component","VLF component", "LF component", "HF component"]

    from matplotlib import rcParams
    plt.rcParams['font.weight'] = 'medium'
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Montserrat']
    matplotlib.rc('xtick', labelsize=15)     
    matplotlib.rc('ytick', labelsize=15)
    Font = {"weight" : "medium"}

    # Plot parameters
    plt.figure(figsize=(10, 10))
    plt.xlabel("Frequency (Hz)", fontsize=25, fontdict=Font, labelpad=25)
    plt.ylabel("PSD (s2/ Hz)", fontsize=25, fontdict=Font, labelpad=15)

    if method == "lomb":
        #plt.title("Lomb's periodogram", fontsize=25)
        for band_index, label in zip(frequency_band_index, label_list):
            plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        plt.legend(prop={"size": 15}, loc="best")
        plt.ylim(0, 0.1)

    elif method == "welch":
        #plt.title("FFT Spectrum : Welch's periodogram", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        plt.legend(prop={"size": 15}, loc="best")
        plt.xlim(0, hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")
    

    plt.savefig("static/Temporary/Lomb_Temporary.png")
    return "Successfully saved Lomscargle"