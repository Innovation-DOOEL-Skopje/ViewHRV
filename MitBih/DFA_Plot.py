import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import nolds
import seaborn as sns

plt.ioff()

sns.set(font_scale=1, font="arial")

def dfa(nn=None, rpeaks=None, short=None, long=None, show=True, figsize=None, legend=True):

    # Check input values
    nn = Check_Input(nn, rpeaks)

    # Check intervals
    short = Check_Interval(short, default=(4, 16))
    long = Check_Interval(long, default=(17, 64))

    # Create arrays
    short = range(short[0], short[1] + 1)
    long = range(long[0], long[1] + 1)

    # Prepare plot
    if figsize is None:
        figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title('Detrended Fluctuation Analysis (DFA)', fontsize=25)
    ax.set_xlabel('log n [beats]',fontsize=20)
    ax.set_ylabel('log F(n)',fontsize=20)

    # try:
    # Compute alpha values
    try:
        alpha1, dfa_short = nolds.dfa(nn, short, debug_data=True, overlap=False)
        alpha2, dfa_long = nolds.dfa(nn, long, debug_data=True, overlap=False)
    except ValueError:
        # If DFA could not be conducted due to insufficient number of NNIs, return an empty graph and 'nan' for alpha1/2
        warnings.warn("Not enough NNI samples for Detrended Fluctuations Analysis.")
        ax.axis([0, 1, 0, 1])
        ax.text(0.5, 0.5, '[Insufficient number of NNI samples for DFA]', horizontalalignment='center',
                verticalalignment='center')
        alpha1, alpha2 = 'nan', 'nan'
    else:
        # Plot DFA results if number of NNI were sufficent to conduct DFA
        # Plot short term DFA
        vals, flucts, poly = dfa_short[0], dfa_short[1], np.polyval(dfa_short[2], dfa_short[0])
        label = r'$ \alpha_{1}: %0.2f$' % alpha1
        ax.plot(vals, flucts, 'bo', markersize=1)
        ax.plot(vals, poly, 'b', label=label, alpha=0.7)

        # Plot long term DFA
        vals, flucts, poly = dfa_long[0], dfa_long[1], np.polyval(dfa_long[2], dfa_long[0])
        label = r'$ \alpha_{2}: %0.2f$' % alpha2
        ax.plot(vals, flucts, 'go', markersize=1)
        ax.plot(vals, poly, 'g', label=label, alpha=0.7)

        # Add legend
        if legend:
            ax.legend()
        ax.grid()

    # Plot axis
    # if show:
    #     plt.show()

    # Output
    # args = (fig, alpha1, alpha2, short, long)
    # names = ('dfa_plot', 'dfa_alpha1', 'dfa_alpha2', 'dfa_alpha1_beats', 'dfa_alpha2_beats')

    fig.savefig("static/Temporary/DFA_Temporary.png")
    # Detrended_Fluctuation_Calculations = dict(zip(args, names))
    plt.close()
    return "Succesfully Saved DFA Plot"

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

def _check_limits(interval, name):

	# upper limit < 0 or upper limit > max interval -> set upper limit to max
	if interval[0] > interval[1]:
		interval[0], interval[1] = interval[1], interval[0]
		vals = (name, name, interval[0], interval[1])
		warnings.warn("Corrected invalid '%s' limits (lower limit > upper limit).'%s' set to: %s" % vals)
	if interval[0] == interval[1]:
		raise ValueError("'%f': Invalid interval limits as they are equal." % name)
	return interval

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