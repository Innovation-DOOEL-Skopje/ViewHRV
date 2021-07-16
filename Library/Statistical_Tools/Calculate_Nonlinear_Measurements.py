import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import nolds

from Library.utils import Check_Input
from Library.utils import Check_Interval
from Library.utils import nn_intervals
from Library.utils import nn_format
from Library.utils import _check_limits
from typing import List

def Calculate_Poincare_Features(nn_intervals: List[float]) -> dict:
    diff_nn_intervals = np.diff(nn_intervals)
    # measures the width of poincare cloud
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)*1000
    # measures the length of the poincare cloud
    sd2 = np.sqrt(2 * np.std(nn_intervals, ddof=1) ** 2 - 0.5 * np.std(diff_nn_intervals, ddof=1) ** 2)*1000
    ratio_sd2_sd1 = sd2 / sd1

    poincare_plot_features = {
        'sd1': sd1,
        'sd2': sd2,
        'ratio_sd2_sd1': ratio_sd2_sd1
    }

    return poincare_plot_features

def Calculate_DFA_Features(nn=None, rpeaks=None, short=None, long=None, show=True, figsize=None, legend=True):

    # Check input values
    nn = Check_Input(nn, rpeaks)

    # Check intervals
    short = Check_Interval(short, default=(4, 16))
    long = Check_Interval(long, default=(17, 64))

    # Create arrays
    short = range(short[0], short[1] + 1)
    long = range(long[0], long[1] + 1)

    # try:
    # Compute alpha values
    try:
        alpha1, dfa_short = nolds.dfa(nn, short, debug_data=True, overlap=False)
        alpha2, dfa_long = nolds.dfa(nn, long, debug_data=True, overlap=False)
    except ValueError:
        # If DFA could not be conducted due to insufficient number of NNIs, return an empty graph and 'nan' for alpha1/2
        warnings.warn("Not enough NNI samples for Detrended Fluctuations Analysis.")
        alpha1 = np.nan
        alpha2 = np.nan
    # Output
    args = (float(alpha1), float(alpha2), str(short), str(long))
    names = ('dfa_alpha1', 'dfa_alpha2', 'dfa_alpha1_beats', 'dfa_alpha2_beats')

    Detrended_Fluctuation_Calculations = {
        "dfa_alpha1" : alpha1,
        "dfa_alpha2" : alpha2,
        # "dfa_alpha1_beats" : short,
        # "dfa_alpha2_beats" : long
    }

    return Detrended_Fluctuation_Calculations