from collections import namedtuple
from typing import List
from typing import Tuple
from astropy.timeseries import LombScargle


# Lomb Scargle Tools 
from Library.utils import _get_features_from_psd
from Library.utils import _get_freq_psd_from_nn_intervals
from Library.utils import _create_timestamp_list

# Named Tuple for different frequency bands
UlfBand = namedtuple("Ulf_band", ["low", "high"])
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])

# Frequency Methods name
WELCH_METHOD = "welch"
LOMB_METHOD = "lomb"

def Calculate_Lomb_Scargle(nn_intervals: List[float], method: str = LOMB_METHOD,
                                  sampling_frequency: int = 4, interpolation_method: str = "linear",
                                  ulf_band: namedtuple = UlfBand(0, 0.00333),
                                  vlf_band: namedtuple = VlfBand(0.00334, 0.04),
                                  lf_band: namedtuple = LfBand(0.04, 0.15),
                                  hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:

    # ----------  Compute frequency & Power spectral density of signal  ---------- #
    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method,
                                                ulf_band=ulf_band,
                                                vlf_band=vlf_band, hf_band=hf_band)

    # ---------- Features calculation ---------- #
    freqency_domain_features = _get_features_from_psd(freq=freq, psd=psd,
                                                      ulf_band=ulf_band,
                                                      vlf_band=vlf_band,
                                                      lf_band=lf_band,
                                                      hf_band=hf_band)

    return freqency_domain_features