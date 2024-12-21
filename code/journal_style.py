import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


class EnableJournalStylePlotting:
    """context manager for plotting graphs with journal style
    """
    def __init__(self):
        self._mpl_context = mpl.rc_context()

    def __enter__(self):
        self._mpl_context.__enter__()

        mpl.rcParams['figure.dpi'] = 100 # 300
        mpl.rcParams['font.family'] = 'DejaVu Serif'
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.markersize'] = 12
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24
        mpl.rcParams['legend.fontsize'] = 24
        mpl.rcParams['axes.titlesize'] = 36
        mpl.rcParams['axes.labelsize'] = 24
        mpl.rcParams['figure.titlesize'] = 36
        mpl.rcParams['figure.labelsize'] = 24

    def __exit__(self, *args, **kwargs):
        self._mpl_context.__exit__(*args, **kwargs)