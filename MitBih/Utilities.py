import wfdb
import numpy as np 
import pandas as pd



def Load_Data(Patient_ID):
    # record = wfdb.rdrecord('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID) + '', )
    annotation = wfdb.rdann('mitdb/mit-bih-arrhythmia-database-1.0.0/' + str(Patient_ID) + '', 'atr',)

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
    # print(NN_Intervals)
    # print(NN_Intervals.size)

    return NN_Intervals


def Load_Data_ANNFILE(ANN):
    
    annotation = pd.read_csv( "//Uploads//" + str(ANN) + ".ann", index_col=False, names=["Peaks", "Symbols"])
    Arr = np.array(annotation.Symbols)

    Table = pd.DataFrame(columns=["RR Peaks", "RR Intervals", "Symbols", "Mask", "Elements_To_Skip", "To_Skip_NN50"])

    Table["RR Peaks"]  = np.array(annotation.Peaks)

    Table["RR Intervals"] = np.insert(np.diff(annotation.Peaks)/360, 0, annotation.Peaks[0], axis=0)

    Table["Symbols"] = annotation.Symbols

    if np.unique(Arr).size > 1:
        None_N_Symbols = list(np.unique(Arr))
        try:
            None_N_Symbols.remove("N")
        except:
            print("No N annotations")
        try:
            None_N_Symbols.remove("+")
        except:
            print("No + annotations")
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
        NN_Intervals_NN50 = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals"])[2:]
        NN_Peaks = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["RR Peaks"]][2:])

        NN_Intervals_Table["RR Intervals Differences"] = np.insert(np.diff(np.array(NN_Intervals_Table["RR Intervals"])), 0, NN_Intervals_Table.loc[0, "RR Intervals"], axis=0)
        RR_Intervals_Differences = np.array(NN_Intervals_Table["RR Intervals Differences"])[2:]

        NN_Intervals_Table["RR Intervals Differences Squared"] = np.array(NN_Intervals_Table["RR Intervals Differences"])**2
        RR_Intervals_Differences_Squared = np.array(NN_Intervals_Table["RR Intervals Differences Squared"])[2:]
    else:
        Table["Mask"] = 0
        Table["Elements_To_Skip"] = 0
        Table["To_Skip_NN50"] = 0

        NN_Intervals_Table = Table.loc[(Table["Mask"] == 0) & (Table["Elements_To_Skip"] == 0)]
        NN_Intervals = np.array(NN_Intervals_Table["RR Intervals"])[2:]
        NN_Intervals_NN50 = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals"])[2:]
        NN_Peaks = np.array(NN_Intervals_Table["RR Peaks"][2:])

        NN_Intervals_Table["RR Intervals Differences"] = np.insert(np.diff(np.array(NN_Intervals_Table["RR Intervals"])), 0, NN_Intervals_Table.loc[0, "RR Intervals"], axis=0)
        RR_Intervals_Differences = np.array(NN_Intervals_Table["RR Intervals Differences"])[2:]

        NN_Intervals_Table["RR Intervals Differences Squared"] = np.array(NN_Intervals_Table["RR Intervals Differences"])**2
        RR_Intervals_Differences_Squared = np.array(NN_Intervals_Table["RR Intervals Differences Squared"])[2:]
    
    NN_Collection = {"ID_Date_Ann": str(ANN), "NN_Intervals" : NN_Intervals, "NN_Intervals_NN50" : NN_Intervals_NN50, "NN_Peaks" : NN_Peaks, "RR_Intervals_Differences" : RR_Intervals_Differences, "RR_Intervals_Differences_Squared" : RR_Intervals_Differences_Squared}
    
    return NN_Collection