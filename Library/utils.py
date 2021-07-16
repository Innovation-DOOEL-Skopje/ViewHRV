import pandas as pd
import numpy as np
import os
import warnings


from collections import namedtuple
from typing import List
from typing import Tuple
from astropy.timeseries import LombScargle

# Named Tuple for different frequency bands
UlfBand = namedtuple("Ulf_band", ["low", "high"])
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])

# Frequency Methods name
WELCH_METHOD = "welch"
LOMB_METHOD = "lomb"


Home_Directory = os.getcwd()





#=================================================================================================================================================================================
# lOADING DATA TOOLS STARTS
def Load_Data_FDBFILES(ID,Date,ANN):
    
    annotation = pd.read_csv( str(Home_Directory) + "\\FDBFILES\\" + str(ID) + "\\" + str(Date) + "\\ANNFILES\\" + str(ANN) + ".ann", index_col=False, names=["Peaks", "Symbols"])
    Arr = np.array(annotation.Symbols)

    Table = pd.DataFrame(columns=["RR Peaks", "RR Intervals", "Symbols", "Mask", "Elements_To_Skip", "To_Skip_NN50"])

    Table["RR Peaks"]  = np.array(annotation.Peaks)

    Table["RR Intervals"] = np.insert(np.diff(annotation.Peaks)/360, 0, annotation.Peaks[0], axis=0)

    Table["Symbols"] = annotation.Symbols


    None_N_Symbols = list(np.unique(Arr))
    try:
        None_N_Symbols.remove("N")
    except:
        pass
    try:
        None_N_Symbols.remove("+")
    except:
        pass
    try:
        None_N_Symbols.remove("L")
    except:
        pass
    try:
        None_N_Symbols.remove("R")
    except:
        pass

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
    NN_Intervals_NN50 = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals"])[2:]
    NN_Peaks = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["RR Peaks"]][2:])

    NN_Intervals_Table["RR Intervals Differences"] = np.insert(np.diff(np.array(NN_Intervals_Table["RR Intervals"])), 0, NN_Intervals_Table.loc[0, "RR Intervals"], axis=0)
    RR_Intervals_Differences = np.array(NN_Intervals_Table["RR Intervals Differences"])[2:]

    NN_Intervals_Table["RR Intervals Differences Squared"] = np.array(NN_Intervals_Table["RR Intervals Differences"])**2
    RR_Intervals_Differences_Squared = np.array(NN_Intervals_Table["RR Intervals Differences Squared"])[2:]

    
    NN_Collection = {"ID_Date_Ann": str(ID) + "_" + str(Date) + "_" + str(ANN), "NN_Itervals" : NN_Intervals, "NN_Intervals_NN50" : NN_Intervals_NN50, "NN_Peaks" : NN_Peaks, "RR_Intervals_Differences" : RR_Intervals_Differences, "RR_Intervals_Differences_Squared" : RR_Intervals_Differences_Squared}
    
    return NN_Collection



def Load_Data_ANNFILE(ANN, Frequency):

    annotation = pd.read_csv("Uploads//" + str(ANN) + ".ann", index_col=False, error_bad_lines = False, engine="python", names=["Peaks", "Symbols"])
    if pd.isnull(annotation).all()["Symbols"] == True:
        Symbol = "N"
        annotation["Symbols"] = Symbol
    else:
        pass
        
    Arr = np.array(annotation.Symbols)

    Table = pd.DataFrame(columns=["RR Peaks", "RR Intervals", "Symbols", "Mask", "Elements_To_Skip", "To_Skip_NN50"])

    Table["RR Peaks"]  = np.array(annotation.Peaks)

    Table["RR Intervals"] = np.insert(np.diff(annotation.Peaks)/Frequency, 0, annotation.Peaks[0], axis=0)

    Table["Symbols"] = annotation.Symbols

    if np.unique(Arr).size > 1:
        None_N_Symbols = list(np.unique(Arr))
        try:
            None_N_Symbols.remove("N")
        except:
            pass
        try:
            None_N_Symbols.remove("+")
        except:
            pass
        try:
            None_N_Symbols.remove("L")
        except:
            pass
        try:
            None_N_Symbols.remove("R")
        except:
            pass
        try:
            None_N_Symbols.remove("B")
        except:
            pass

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
            del Temp
            del Temp_1
            del Temp_2

        Table.loc[Mask, "Mask"] = 1
        Temp = np.delete(np.array(Table.index), Mask)
        Table.loc[Temp, "Mask"] = 0
        del Temp
        
        try:
            Elements_To_Skip = np.delete(Elements_To_Skip, np.argwhere(Elements_To_Skip == Elements_To_Skip.max()))
        except:
            pass
        Table.loc[Elements_To_Skip, "Elements_To_Skip"] = 1
        Temp = np.delete(np.array(Table.index), Elements_To_Skip)
        Table.loc[Temp, "Elements_To_Skip"] = 0
        del Temp
        
        try:
            To_Skip_NN50 = np.delete(To_Skip_NN50, np.argwhere(To_Skip_NN50 == To_Skip_NN50.max()))
            To_Skip_NN50 = np.delete(To_Skip_NN50, np.argwhere(To_Skip_NN50 == To_Skip_NN50.max()))
        except:
            pass
        Table.loc[To_Skip_NN50, "To_Skip_NN50"] = 1
        Temp = np.delete(np.array(Table.index), To_Skip_NN50)
        Table.loc[Temp, "To_Skip_NN50"] = 0
        del Temp

        NN_Intervals_Table = Table.loc[(Table["Mask"] == 0) & (Table["Elements_To_Skip"] == 0)]
        NN_Intervals = np.array(NN_Intervals_Table["RR Intervals"])[2:]
        NN_Intervals_NN50 = np.array(NN_Intervals_Table.loc[NN_Intervals_Table["To_Skip_NN50"] == 0]["RR Intervals"])[2:]
        NN_Peaks = np.array([NN_Intervals_Table["RR Peaks"]])
        NN_Intervals_Table.reset_index(drop=True, inplace=True)
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
        NN_Peaks = np.array([NN_Intervals_Table["RR Peaks"]])

        NN_Intervals_Table["RR Intervals Differences"] = np.insert(np.diff(np.array(NN_Intervals_Table["RR Intervals"])), 0, NN_Intervals_Table.loc[0, "RR Intervals"], axis=0)
        RR_Intervals_Differences = np.array(NN_Intervals_Table["RR Intervals Differences"])[2:]

        NN_Intervals_Table["RR Intervals Differences Squared"] = np.array(NN_Intervals_Table["RR Intervals Differences"])**2
        RR_Intervals_Differences_Squared = np.array(NN_Intervals_Table["RR Intervals Differences Squared"])[2:]
    
    NN_Collection = {"ID_Date_Ann": str(ANN), "NN_Intervals" : NN_Intervals, "NN_Intervals_NN50" : NN_Intervals_NN50, "NN_Peaks" : NN_Peaks, "RR_Intervals_Differences" : RR_Intervals_Differences, "RR_Intervals_Differences_Squared" : RR_Intervals_Differences_Squared}
    
    return NN_Collection, Table

# lOADING DATA TOOLS ENDS
#=================================================================================================================================================================================

#=================================================================================================================================================================================
# lOMB SCARGLE TOOLS STARTS

def _create_timestamp_list(nn_intervals: List[float]) -> List[float]:

    # Convert in seconds
    nni_tmstp = np.cumsum(nn_intervals) / 1000

    # Force to start at 0
    return nni_tmstp - nni_tmstp[0]

def _get_freq_psd_from_nn_intervals(nn_intervals: List[float], method: str = LOMB_METHOD,
                                    sampling_frequency: int = 4,
                                    interpolation_method: str = "linear",
                                    ulf_band: namedtuple = UlfBand(0, 0.00333),
                                    vlf_band: namedtuple = VlfBand(0.00334, 0.04),
                                    hf_band: namedtuple = HfBand(0.15, 0.40)) -> Tuple:

    timestamp_list = _create_timestamp_list(nn_intervals)



    if method == LOMB_METHOD:
        freq, psd = LombScargle(timestamp_list, nn_intervals,
                                normalization='psd').autopower(minimum_frequency=ulf_band[0],
                                                               # THIS WHERE I CHANGED VLF_BAND TO ULF_BAND
                                                               maximum_frequency=hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq, psd


def _get_features_from_psd(freq: List[float], psd: List[float], 
                           ulf_band: namedtuple = UlfBand(0, 0.00333),
                           vlf_band: namedtuple = VlfBand(0.00333, 0.04),
                           lf_band: namedtuple = LfBand(0.04, 0.15),
                           hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:

    # Calcul of indices between desired frequency bands
    ulf_indexes = np.logical_and(freq >= ulf_band[0], freq < ulf_band[1])
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    # Integrate using the composite trapezoidal rule
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

    # total power & vlf : Feature often used for  "long term recordings" analysis
    ulf = np.trapz(y=psd[ulf_indexes], x=freq[ulf_indexes])
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    total_power = ulf + vlf + lf + hf

    lf_hf_ratio = lf / hf
    lfnu = (lf / (lf + hf)) * 100
    hfnu = (hf / (lf + hf)) * 100

    freqency_domain_features = {
        'lf': str(lf),
        'hf': str(hf),
        'lf_hf_ratio': str(lf_hf_ratio),
        'lfnu': str(lfnu),
        'hfnu': str(hfnu),
        'total_power': str(total_power),
        'vlf': str(vlf),
        'ulf' : str(ulf)
    }

    return freqency_domain_features

# lOMB SCARGLE TOOLS ENDS
#=================================================================================================================================================================================

# NONLINEAR MEASUREMENTS STARTS
#=================================================================================================================================================================================

def nn_intervals(rpeaks=None):

	# Check input signal
	if rpeaks is None:
		raise TypeError("No data for R-peak locations provided. Please specify input data.")
	elif type(rpeaks) is not list and not np.ndarray:
		raise TypeError("List, tuple or numpy array expected, received  %s" % type(rpeaks))

	# if all(isinstance(n, int) for n in rpeaks) is False or all(isinstance(n, float) for n in rpeaks) is False:
	# 	raise TypeError("Incompatible data type in list or numpy array detected (only int or float allowed).")

	# Confirm numpy arrays & compute NN intervals
	rpeaks = np.asarray(rpeaks)
	nn_int = np.zeros(rpeaks.size - 1)

	for i in range(nn_int.size):
		nn_int[i] = rpeaks[i + 1] - rpeaks[i]

	return nn_format(nn_int)

def nn_format(nni=None):

	# Check input
	if nni is None:
		raise TypeError("No input data provided for 'nn'. Please specify input data")
	nn_ = np.asarray(nni, dtype='float64')

	# Convert if data has been identified in [s], else proceed with ensuring the NumPy array format
	if np.max(nn_) < 10:
		nn_ = [int(x * 1000) for x in nn_]

	return np.asarray(nn_)


def Check_Input(nni=None, rpeaks=None):
	# Check input
	if nni is None and rpeaks is not None:
		# Compute NN intervals if r_peaks array is given
		nni = nn_intervals(rpeaks)
	elif nni is not None:
		# Use given NN intervals & confirm numpy
		nni = nn_format(nni)
	else:
		raise TypeError("No R-peak data or NN intervals provided. Please specify input data.")
	return nni


def _check_limits(interval, name):

	# upper limit < 0 or upper limit > max interval -> set upper limit to max
	if interval[0] > interval[1]:
		interval[0], interval[1] = interval[1], interval[0]
		vals = (name, name, interval[0], interval[1])
		warnings.warn("Corrected invalid '%s' limits (lower limit > upper limit).'%s' set to: %s" % vals)
	if interval[0] == interval[1]:
		raise ValueError("'%f': Invalid interval limits as they are equal." % name)
	return interval

def Check_Interval(interval=None, limits=None, default=None):

	if interval is None and limits is None and default is None:
		raise TypeError("No input data specified. Please verify your input data.")

	# Create local copy to prevent changing input variable
	interval = list(interval) if interval is not None else None

	# Check default limits
	if default is not None:
		default = _check_limits(default, 'default')

	# Check maximum range limits
	if limits is None and default is not None:
		limits = default
	elif limits is not None:
		limits = _check_limits(limits, 'limits')

	# Check interval limits
	if interval is None:
		if default is not None:
			return default
		elif default is None and limits is not None:
			return limits

	# If only interval is specified, but not 'min', 'max' or 'default' check if lower limit >= upper limit
	elif interval is not None and limits is None:
		return _check_limits(interval, 'interval')

	# If none of the input is 'None'
	else:
		# Check interval
		interval = _check_limits(interval, 'interval')
		if not limits[0] <= interval[0]:
			interval[0] = limits[0]
			warnings.warn("Interval limits out of boundaries. Interval set to: %s" % interval, stacklevel=2)
		if not limits[1] >= interval[1]:
			interval[1] = limits[1]
			warnings.warn("Interval limits out of boundaries. Interval set to: %s" % interval, stacklevel=2)
		return interval


# NONLINEAR MEASUREMENTS ENDS
#=================================================================================================================================================================================