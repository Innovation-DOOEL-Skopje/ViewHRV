# Necessery Imports
import numpy as np

# This functions takes  a numpy array of the NN intervals, the variable is created through loading the data from the Load_Data function where
# which returns a dict containing the key NN_Intervals with its corresponding variables. It calculates the standard deviation and return the value.
def Calculate_SDNN(NN_Intervals):

    SDNN = np.std(NN_Intervals)*1000
    return SDNN


# This function takes a numpy array NN_Intervals, splits the array into six segments and then averages the standard deviations of those segments.
def Calculate_ASDNN(NN_Intervals):

    try:
        Base = round(len(NN_Intervals)/6)
        S1 = NN_Intervals[0:Base*1]
        S2 = NN_Intervals[Base*1:Base*2]
        S3 = NN_Intervals[Base*2:Base*3]
        S4 = NN_Intervals[Base*3:Base*4]
        S5 = NN_Intervals[Base*4:Base*5]
        S6 = NN_Intervals[Base*5:]
        ASDNN = np.average([np.std(S1), np.std(S2), np.std(S3), np.std(S4), np.std(S5), np.std(S6)])*1000
    except:
        Base = round(len(NN_Intervals)/5)
        S1 = NN_Intervals[0:Base*1]
        S2 = NN_Intervals[Base*1:Base*2]
        S3 = NN_Intervals[Base*2:Base*3]
        S4 = NN_Intervals[Base*3:Base*4]
        S5 = NN_Intervals[Base*4:]
        ASDNN = np.average([np.std(S1), np.std(S2), np.std(S3), np.std(S4), np.std(S5), np.std(S6)])*1000

    return ASDNN


# This function takes a numpy array NN_Intervals, splits the array into seven segments then calculates their averages after that calculates the average of all seven averages
# and finally calculates the square root of the sum of differences betqween each average with the mean of the averages.
def Calculate_SDANN(NN_Intervals):

    Base = round(len(NN_Intervals)/7)

    S1 = NN_Intervals[0:Base*1]
    S2 = NN_Intervals[Base*1:Base*2]
    S3 = NN_Intervals[Base*2:Base*3]
    S4 = NN_Intervals[Base*3:Base*4]
    S5 = NN_Intervals[Base*4:Base*5]
    S6 = NN_Intervals[Base*5:Base*6]
    S7 = NN_Intervals[Base*6:]
    
    print(NN_Intervals)
    
    print(S1)
    
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

    Differences = (Averages-NN_Mean)
    Differences

    SDANN = np.sqrt(np.sum(Differences**2)/Differences.size)*1000

    return SDANN



# This function takes the Differences between RR peaks and sees which readings are greater than 0.00251. Since we are seeing what reading are greater than 50ms by
# squaring our values we must square the ms aswell.
def Calculate_NN50(RR_Intervals_Differences_Squared):

    Differences_Squared = RR_Intervals_Differences_Squared

    NN50 = (Differences_Squared[1:] > 0.00251)
    NN50 = Differences_Squared[1:][NN50].size

    return NN50


# Calculates the percentage of NN50 from the total number of beats.
def Calculate_pNN50(NN_Intervals, NN50):

    pNN50 = (np.sum(NN50) / NN_Intervals.size)*100

    return pNN50


# Calculates the square root of the sum of differences squared over the total size.
def Calculate_rMSSD(NN_Intervals):

    rMSSD = np.sqrt(np.sum(np.diff(NN_Intervals)**2)/np.diff(NN_Intervals).size)*1000

    return rMSSD



# Calculates all the indices necessary and then return a dictionary element.
def Calculate_Time_Domain(NN_Intervals, RR_Intervals_Differences_Squared):
    
    SDNN = round(Calculate_SDNN(NN_Intervals), 3)
    ASDNN = round(Calculate_ASDNN(NN_Intervals), 3)
    SDANN = round(Calculate_SDANN(NN_Intervals), 3)
    NN50 = round(Calculate_NN50(RR_Intervals_Differences_Squared), 3)
    pNN50 = round(Calculate_pNN50(NN_Intervals, NN50), 3)
    rMSSD = round(Calculate_rMSSD(NN_Intervals), 3)

    Time_Domain_Indices = {"SDNN" : SDNN, "ASDNN" : ASDNN, "SDANN" : SDANN, "NN50" : NN50, "pNN50" : pNN50, "rMSSD" : rMSSD}

    return Time_Domain_Indices