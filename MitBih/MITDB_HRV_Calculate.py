import pandas as pd
import numpy as np
import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wfdb.processing.hr import calc_rr
import json
# import pyhrv.time_domain as td
# import pyhrv.frequency_domain as fd
# import pyhrv.nonlinear as nl
from IPython.display import clear_output
import warnings
import pickle
import math

plt.ioff()

warnings.filterwarnings("ignore")
sns.set()

def Calculate_HRV(Patient_ID):
    # record = wfdb.rdrecord('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID) + '', )
    annotation = wfdb.rdann('./mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID) + '', 'atr',)

#     print(len(annotation.sample))
#     print(len(annotation.symbol))

    Arr = annotation.symbol
#     print(Arr[:10])
#     print(len(Arr))

    Table = pd.DataFrame(columns=["RR Peaks", "RR Intervals", "Symbols", "Mask", "Elements_To_Skip", "To_Skip_NN50"])

    Table["RR Peaks"]  = annotation.sample

    Table["RR Intervals"] = np.insert(np.diff(annotation.sample)/360, 0, annotation.sample[0], axis=0)

    Table["Symbols"] = annotation.symbol

    None_N_Symbols = list(np.unique(Arr))
    None_N_Symbols.remove("N")
    None_N_Symbols.remove("+")
    None_N_Symbols

    Mask = np.empty(0)
    Elements_To_Skip = np.empty(0)
    To_Skip_NN50 = np.empty(0)
    for S in None_N_Symbols:
        Temp = np.array([i for i, v in enumerate(Arr) if str(S) in v])
        Temp_1 = np.array([i for i, v in enumerate(Arr) if str(S) in v]) + 1
        Temp_2 = np.array([i for i, v in enumerate(Arr) if str(S) in v]) + 2
        Mask = np.concatenate((Mask, Temp))
        Elements_To_Skip = np.concatenate((Elements_To_Skip, Temp_1))
        To_Skip_NN50 = np.concatenate((To_Skip_NN50, Temp_2))
        print(S)
        del Temp
        del Temp_1
        del Temp_2

    Table[:11]

    Table.loc[Mask, "Mask"] = 1
    Temp = np.delete(np.array(Table.index), Mask)
    Table.loc[Temp, "Mask"] = 0
    del Temp

    Table.loc[Elements_To_Skip, "Elements_To_Skip"] = 1
    Temp = np.delete(np.array(Table.index), Elements_To_Skip)
    Table.loc[Temp, "Elements_To_Skip"] = 0
    del Temp

    Table.loc[To_Skip_NN50, "To_Skip_NN50"] = 1
    Temp = np.delete(np.array(Table.index), To_Skip_NN50)
    Table.loc[Temp, "To_Skip_NN50"] = 0
    del Temp

    NN_Intervals_Table = Table.loc[(Table["Mask"] == 0) & (Table["Elements_To_Skip"] == 0)]

    NN_Intervals = np.array(NN_Intervals_Table["RR Intervals"])[2:]
#     print(NN_Intervals)
#     print(NN_Intervals.size)

    NN_Intervals_NN50 = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals"])[2:]
#     print(NN_Intervals_NN50)
#     print(NN_Intervals_NN50.size)

    # Calculate Time Domain

    HRV = {"Patient_ID" : Patient_ID, "Time_Domain" : {}, "Frequency_Domain" : {}, "NoneLinear_Measurements" : {"Poincare" : {}, "DFA" : {}}}

    ## SDNN

    SDNN = np.std(NN_Intervals)*1000
    SDNN

    HRV["Time_Domain"]["SDNN"] = round(SDNN,3)
    HRV

    ## ASDNN

    Base = round(len(NN_Intervals)/6)
    Base

    S1 = NN_Intervals[0:Base*1]
    S2 = NN_Intervals[Base*1:Base*2]
    S3 = NN_Intervals[Base*2:Base*3]
    S4 = NN_Intervals[Base*3:Base*4]
    S5 = NN_Intervals[Base*4:Base*5]
    S6 = NN_Intervals[Base*5:]

    ASDNN = np.average([np.std(S1), np.std(S2), np.std(S3), np.std(S4), np.std(S5), np.std(S6)])*1000
    ASDNN

    HRV["Time_Domain"]["ASDNN"] = round(ASDNN,3)
    HRV

    ## Calculate SDANN

    Base = round(len(NN_Intervals)/7)
    Base

    S1 = NN_Intervals[0:Base*1]
    S2 = NN_Intervals[Base*1:Base*2]
    S3 = NN_Intervals[Base*2:Base*3]
    S4 = NN_Intervals[Base*3:Base*4]
    S5 = NN_Intervals[Base*4:Base*5]
    S6 = NN_Intervals[Base*5:Base*6]
    S7 = NN_Intervals[Base*6:]

    AVG1 = np.average(S1)
    AVG2 = np.average(S2)
    AVG3 = np.average(S3)
    AVG4 = np.average(S4)
    AVG5 = np.average(S5)
    AVG6 = np.average(S6)
    AVG7 = np.average(S7)

    Averages = np.array([AVG1, AVG2, AVG3, AVG4, AVG5, AVG6, AVG7])
    Averages

    NN_Mean = np.average(Averages)
    NN_Mean

    Differences = (Averages-NN_Mean)
    Differences

    SDANN = np.sqrt(np.sum(Differences**2)/Differences.size)*1000
    SDANN

    HRV["Time_Domain"]["SDANN"] = round(SDANN,3)

    ## Calculate NN50

    NN_Intervals_Table.shape

    NN_Intervals_Table[:11]

    NN_Intervals_Table["RR Intervals Differences"] = np.insert(np.diff(np.array(NN_Intervals_Table["RR Intervals"])), 0, NN_Intervals_Table.loc[0, "RR Intervals"], axis=0)

    NN_Intervals_Table["RR Intervals Differences Squared"] = np.array(NN_Intervals_Table["RR Intervals Differences"])**2

    Differences_Squared = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals Differences Squared"])[2:]

    NN50 = (Differences_Squared[1:] > 0.00251)
    NN50 = Differences_Squared[1:][NN50].size
    NN50

    HRV["Time_Domain"]["NN50"] = NN50

    ## pNN50

    pNN50 = (np.sum(NN50) / NN_Intervals.size)*100
    pNN50

    HRV["Time_Domain"]["pNN50"] = round(pNN50,3)
    
    ## rMSSD

    rMSSD = np.sqrt(np.sum(np.diff(NN_Intervals)**2)/np.diff(NN_Intervals).size)*1000

    HRV["Time_Domain"]["rMSSD"] = round(rMSSD,3)

    # Frequency Domian Calculations

    RR_Intervals = Table["RR Intervals"]

    fbands = {'ulf': (0.0, 0.0033), 'vlf': (0.0033, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
    FD = dict(fd.welch_psd(NN_Intervals,show=False, fbands=fbands))

    HRV["Frequency_Domain"]["ULF"] = round(FD["fft_abs"][0],3)
    HRV["Frequency_Domain"]["VLF"] = round(FD["fft_abs"][1],3)
    HRV["Frequency_Domain"]["LF"] = round(FD["fft_abs"][2],3)
    HRV["Frequency_Domain"]["HF"] = round(FD["fft_abs"][3],3)
    HRV["Frequency_Domain"]["LF-HF"] = round(FD["fft_ratio"],3)
    HRV["Frequency_Domain"]["Total_Power"] = round(FD["fft_total"],3)
    HRV
    plt.close()
    # None - Linear Measurements Analysis

    Poincare = nl.poincare(RR_Intervals[1:], show=False)
    plt.close()
    DFA = nl.dfa(NN_Intervals, show=False)

    HRV["NoneLinear_Measurements"]["Poincare"]["SD1"] = round(Poincare["sd1"],3)
    HRV["NoneLinear_Measurements"]["Poincare"]["SD2"] = round(Poincare["sd2"],3)
    HRV["NoneLinear_Measurements"]["Poincare"]["SD_Ratio"] = round(Poincare["sd_ratio"],3)
    HRV["NoneLinear_Measurements"]["Poincare"]["Ellipse_Area"] = round(Poincare["ellipse_area"],3)
    HRV["NoneLinear_Measurements"]["DFA"]["DFA_Alpha_1"] = round(DFA["dfa_alpha1"],3)
    HRV["NoneLinear_Measurements"]["DFA"]["DFA_Alpha_2"] = round(DFA["dfa_alpha2"],3)
    HRV
    plt.close()
    return HRV