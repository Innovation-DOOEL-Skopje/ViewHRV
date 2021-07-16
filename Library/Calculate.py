# Compatibility
from __future__ import absolute_import

# Imports
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import warnings
import json
import os
import sys

# Import Statistical Tools
import Library.Statistical_Tools.Calculate_Time_Domain as td
import Library.Statistical_Tools.Calculate_Frequency_Domain as fd
import Library.Statistical_Tools.Calculate_Nonlinear_Measurements as nd

#Import Ploting Tools
import Library.Ploting_Tools.Lomb_Plot as lp
import Library.Ploting_Tools.Poincare_Plot as pp
import Library.Ploting_Tools.DFA_Plot as dp
import Library.Ploting_Tools.ECG_Plot as ep

# Import Preprocessing Tools
# import Preprocess_Data.Edit_Auxann_Files as eaf
# import Preprocess_Data.Merge as mg
# import Preprocess_Data.Quality_Check_Final as qcf 

# Import Utility Tools
import Library.utils as us



warnings.filterwarnings("ignore")


def Calculate(ANN, Frequency):

    # Read the ann file given and return a dictionary matching the below format
    # NN_Collection = 
    # {
    # "ID_Date_Ann": str(ANN), "NN_Intervals" : NN_Intervals, 
    # "NN_Intervals_NN50" : NN_Intervals_NN50, "NN_Peaks" : NN_Peaks, 
    # "RR_Intervals_Differences" : RR_Intervals_Differences, 
    # "RR_Intervals_Differences_Squared" : RR_Intervals_Differences_Squared
    # }
    ANN, Table = us.Load_Data_ANNFILE(str(ANN), Frequency)


    Time_Domain_Features = td.Calculate_Time_Domain(ANN["NN_Intervals"], ANN["RR_Intervals_Differences_Squared"])
    Frequency_Domain_Features = fd.Calculate_Lomb_Scargle(ANN["NN_Intervals"]*1000, method = "lomb")
    Poincare_Features = nd.Calculate_Poincare_Features(ANN["NN_Intervals"])
    DFA_Features = nd.Calculate_DFA_Features(ANN["NN_Intervals"])


    HRV = {"Time_Domain_Features" : Time_Domain_Features, 
            "Frequency_Domain_Features" : Frequency_Domain_Features,    
            "Nonlinear_Measurements" : {"Poincare_Features" : Poincare_Features, 
                                        "DFA_Features" : DFA_Features},
            }

    print(lp.Psd_Plot(ANN["NN_Intervals"]*1000, method="lomb"))
    print(pp.Poincare_Plot(ANN["NN_Intervals"]*1000))
    print(dp.DFA_Plot(ANN["NN_Intervals"]*1000))
    print(ep.RR_Time_Series(Table["RR Intervals"][1:]))

    return HRV

if __name__== "__main__":
    Calculate(sys.argv[1], sys.argv[2]) 