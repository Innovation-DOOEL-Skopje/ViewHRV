import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import nolds
import seaborn as sns
import Library.utils as uti

plt.ioff()

sns.set(font_scale=1, font="arial")

def DFA_Plot(nn=None, rpeaks=None, short=None, long=None, show=True, figsize=None, legend=True):

    # Check input values
    nn = uti.Check_Input(nn, rpeaks)

    # Check intervals
    short = uti.Check_Interval(short, default=(4, 16))
    long = uti.Check_Interval(long, default=(17, 64))

    # Create arrays
    short = range(short[0], short[1] + 1)
    long = range(long[0], long[1] + 1)

    # Prepare plot
    if figsize is None:
        figsize = (10, 10)
    from matplotlib import rcParams
    plt.rcParams['font.weight'] = 'medium'
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Montserrat']
    matplotlib.rc('xtick', labelsize=15)     
    matplotlib.rc('ytick', labelsize=15)
    Font = {"weight" : "medium"}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    #ax.set_title('Detrended Fluctuation Analysis (DFA)', fontsize=25)
    ax.set_xlabel('log n [beats]',fontsize=25, fontdict=Font, labelpad=25)
    ax.set_ylabel('log F(n)',fontsize=25, fontdict=Font, labelpad=25)

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