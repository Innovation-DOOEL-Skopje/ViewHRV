import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

matplotlib.use("agg")
plt.ioff()

sns.set(font_scale=2, font="arial")


def ECG(Patient_ID):
    record = wfdb.rdrecord('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID), sampto=10000)
    annotation = wfdb.rdann('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID), 'atr', sampto=10000)
    Plot = wfdb.plot_wfdb(record, annotation=annotation,ann_style=["r."], time_units="seconds", figsize=(40,40), return_fig=True, title="ECG Signal for Every Channel")
    Plot.savefig("static/Temporary/ECG_Temporary.png")
    plt.close()

    return "Successfully Saved ECG Plot"