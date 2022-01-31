import numpy as np
import quickspikes as qs
import peakutils.peak
import matplotlib.pyplot as plt
from scipy.signal import lfilter, savgol_filter
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
import ipywidgets as widgets
from ipywidgets import interact, interactive
from IPython.display import display, clear_output

def IIR_smoothing(plot_x, plot_y, n):
    '''
    smoothing noisy data points by IIR filter, smoothing parameter controllable by interactive widget

    Args:

    plot_x: time points, plotted on the x-axis 

    plot_y: resonance units corresponding to the time points; plotted on the y-axis

    n: the larger the value of n, the smoother the resultant curve will be

    Returns:

    y_filtered: output of the digitial IIR filter

    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.plot(plot_x, plot_y, '-*')
    b = [1.0 / n] * n
    a = 1
    y_filtered = lfilter(b, a, plot_y)
    p, = ax.plot(plot_x, y_filtered, 'r')
    # Properties of the Slider widget button
    win_size = widgets.IntSlider(value = n, min = 1, max = 500, step = 1, description='IIR Coefficient: ')
    fig.set_figwidth(10)
    fig.set_figheight(10)

    def update(val):
        new_n = int(win_size.value)
        new_b = [1.0 / new_n] * new_n
        new_y = lfilter(new_b, a, plot_y)
        p.set_ydata(new_y)
        y_filtered = new_y
        display(fig)
        clear_output(wait = True) 
        plt.pause(0.1)
    
    interact(update, val=win_size)
    return y_filtered

def SVG_filter(plot_x, plot_y, win_len, polyorder, **kwargs):
    '''
    smoothing noisy data points by applying a Savitzky-Golay filter, smoothing parameter controllable by interactive widget

    Args:

    plot_x: time points, plotted on the x-axis 

    plot_y: resonance units corresponding to the time points; plotted on the y-axis

    win_len: the length of the filter window; must be a positive odd integer

    polyorder: int, the order of the polynomial used to fit the samples, polyorder must be less than window_length

    **kwargs: other keyword parameters that can be specified for the filter

    Returns:

    y_filtered: output of the low-pass Savitzky-Golay filter

    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.plot(plot_x, plot_y, '-*')
    y_filtered = savgol_filter(plot_y, win_len, polyorder, **kwargs)
    p, = ax.plot(plot_x, y_filtered, 'r')
    # Properties of the Slider widget button
    win_size = widgets.IntSlider(value = win_len, min = 5, max = 499, step = 2, description='SVG Window size: ')
    fig.set_figwidth(10)
    fig.set_figheight(10)

    def update(val):
        new_win = int(win_size.value)
        new_y = savgol_filter(plot_y, new_win, polyorder)
        p.set_ydata(new_y)
        y_filtered = new_y
        display(fig)
        clear_output(wait = True) 
        plt.pause(0.1)

    interact(update, val=win_size)
    return y_filtered

def peak_detection_scipy(signal, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
    '''
    find peaks inside a signal based on peak properties
    this function takes a 1-D array and finds all local maxima by simple comparison of neighboring values
    optionally, a subset of these peaks can be selected by specifying conditions for a peak's properties

    Args:

    signal: number or ndarray or equence; a signal with peaks

    height: number or ndarray or sequence, optional; required height of peaks

    threshold: sequence, optional; required threshold of peaks, the vertical distance to its neighboring samples

    distance: number, optional; required minimal horizontal distance (>= 1) in samples between neighbouring peaks

    prominence: number or ndarray or sequence, optional; required prominence of peaks, which is the minimum height necessary to descend
    to get from the summit to any higher terrain. the higher the prominence, the more "important" the peak is

    width: number or ndarray or sequence, optional; required width of peaks in samples

    wlen: int, optional; used for calculation of the peaks prominences, a window length in samples that optionally
    limits the evaluated area for each peak to a subset of the signal

    rel_height: float, optional; used for calculation of the peaks width, calculates the relative height at which
    the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its
    lowest contour line while 0.5 evaluates at half the prominence length. Must be at least 0.

    plateau_size: number or ndarray or sequence, optional; required size of the flat top of peaks in samples

    Returns:

    peaks: ndarray; indices of peaks in signal that satisfy all the specified arguments/conditions

    properties: dict; a dictionary containing properties of the returned peaks which were calculated as intermediate
    results during evaluation of the specified arguments/conditions

    '''
    peaks, properties = find_peaks(signal, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
    return peaks, properties

def peakutils_detection(y, thres=0.3, min_dist=1, thres_abs=False):
    '''
    peak detection routine finds the numeric index of the peaks in y by taking its first order difference
    by using thres and min_dist parameters, and it is possible to reduce the number of detected peaks
    
    Args:

    y: ndarray (signed); 1D amplitude data to search for peaks
    
    thres: float between [0,1], optional; normalized threshold and only the peaks with amplitude higher than thres
    will be detected

    min_dist: int, optional; minimum distance between each detected peak. the peak with the higest amplitude
    is preferred to satisfy this constraint if specified.

    thres_abs: boolean, optional; if True, the thres value will be interpreted as an absolute value instead of a normalized threshold

    Returns:

    peaks: ndarray; array containing the numeric indexes of the peaks that were detected

    '''
    peaks = peakutils.peak.indexes(y, thres, min_dist, thres_abs)
    return peaks

def spike_removal(samples, times, thresh_v, thresh_dv, min_size):
    """
    Spikes are removed from the time series by beginning at each peak and
    moving in either direction until signal drops below thresh_v OR dv drops below
    thresh_dv. This algorithm does not work with negative peaks, so invert your
    signal if you need to. (inversion can be done by multiplying signal with -1)

    Args:

    samples: signal to analyze
    
    times: times of the peaks (in samples)
    
    thresh_v: number; do not include points contiguous with the peak with V > thresh_v
    
    thresh_dv: number; do not include points contiguous with the peak with deltaV > thresh_dv.
    negative values correspond to positive-going signals
    
    min_size: int; spike removal always remove at least this many points on either side of peak
    
    Returns:
    
    output: a copy of samples with the data points around spike times set to NaN

    """
    output = qs.subthreshold(samples, times, thresh_v, thresh_dv, min_size)
    return output
