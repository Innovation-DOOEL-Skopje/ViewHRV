import pandas as pd
import numpy as np
import wfdb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import os
from typing import List
from matplotlib import rcParams


matplotlib.use("agg")
plt.ioff()

sns.set(font_scale=2, font="arial")



def ECG_Plot(nn_intervals: List[float], normalize: bool = True,
                    autoscale: bool = True, y_min: float = None, y_max: float = None):

    style.use("seaborn-darkgrid")
    plt.figure(figsize=(21, 11))
    plt.title("RR Interval Time Series")
    plt.ylabel("RR Interval", fontsize=15)

    if normalize:
        plt.xlabel("Time (s)", fontsize=15)
        plt.plot(np.cumsum(nn_intervals) / 1000, nn_intervals)
    else:
        plt.xlabel("RR-Interval index", fontsize=15)
        plt.plot(nn_intervals)

    if not autoscale:
        plt.ylim(y_min, y_max)
    plt.show()

    plt.savefig("static/Temporary/ECG_Temporary.png")
    # ECG
    plt.close()
    return "Succesfully Saved ECG Plot"


def RR_Time_Series(rr_intervals: List[float], normalize: bool = True, autoscale: bool = True, y_min: float = None, y_max: float = None):
    from matplotlib import rcParams
    plt.rcParams['font.weight'] = 'medium'
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Montserrat']
    matplotlib.rc('xtick', labelsize=15)     
    matplotlib.rc('ytick', labelsize=15)
    Font = {"weight" : "medium"}
    plt.figure(figsize=(21, 8))
    plt.ylabel("RR Interval", fontsize=25, fontdict=Font, labelpad=25)

    if normalize:
        plt.xlabel("Time (minutes)", fontsize=25, fontdict=Font, labelpad=15)
        plt.plot(np.cumsum(rr_intervals)/60, rr_intervals)
    else:
        plt.xlabel("RR-Interval index", fontsize=25, fontdict=Font, labelpad=25)
        plt.plot(rr_intervals)

    if not autoscale:
        plt.ylim(y_min, y_max)
    
    plt.show()

    plt.savefig("static/Temporary/ECG_Temporary.png")
    # ECG
    # plt.close()
    return "Succesfully Saved ECG Plot"

def Mit_Bih_ECG(Patient_ID):
    record = wfdb.rdrecord('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID), sampto=10000)
    annotation = wfdb.rdann('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID), 'atr', sampto=10000)
    Plot = wfdb.plot_wfdb(record, annotation=annotation,ann_style=["r."], time_units="seconds", figsize=(40,40), return_fig=True, title="ECG Signal for Every Channel")
    Plot.savefig("static/Temporary/ECG_Temporary.png")
    plt.close()

    return "Successfully Saved ECG Plot"