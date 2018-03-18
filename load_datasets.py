#Script to download datasets and functions to change the datasets.
#Functions:
#1)load_datasets
#2)calculate_moving_mean
#3)normalized_datasets (the standarize equation is applied to all climatic variables and LAI,
# to see which affects lai model the most)
#4)savitzky_goley (filter to smooth the orginal datasets from NASA to avpid some basic errors)

from settings import settings
import h5py
import numpy
import logging
from datetime import date

timsetamps = None

datasets = {
    'lai': None,
    'tmp': None,
    'pre': None,
    'vap': None,
    'pet': None,
}

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def load_data():

    storage_name = settings['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:

        x_time = data_file['timestamps']

        time_x = []
        for t in x_time:
            dt = date.fromtimestamp(t)
            time_x.append(dt)

        global timestamps
        timestamps = time_x[:120]
        groupname = settings['groupname']
        for ds_name in datasets.keys():
            data = list(data_file[f'{groupname}-{ds_name}'])[:120]
            if settings.get('normalize') and ds_name != 'lai':
                log.error("Normalizing %s", ds_name)
                data = normalized_dataset(data)
            datasets[ds_name] = data

    return timestamps, datasets


def calculate_moving_mean():
    """
    :return: a plot.
    """
    moving_avg = []
    ds_var = settings['prediction_option']
    moving_average_result = [0]
    x_months = settings.get('moving_average_months', 0)
    if not x_months:
        log.debug('No moving average defined')
        return

    moving_average_result = (x_months - 1) * [0]

    for value in datasets[ds_var]:
        moving_avg.append(value)

        if len(moving_avg) > x_months:
            # remove oldest value
            moving_avg.pop(0)

        if len(moving_avg) == x_months:
            m_avg = sum(moving_avg) / x_months
            moving_average_result.append(m_avg)

    dataset_label = f'{ds_var}_moving_avg_{x_months}'
    assert len(moving_average_result) == len(datasets[ds_var])
    datasets[dataset_label] = moving_average_result

    #from matplotlib import pyplot
    #pyplot.plot(timestamps[8:], moving_average_result[8:], 'b', timestamps, datasets[ds_var], 'g')
    #pyplot.show()


def normalized_dataset(source_data):
    """
    Apply  Z(x(i))= {x(i)-avg(x)}/sd(x) to given data set. Standardize all values to +/- Standard deviations.
    Mean is everywhere 0.
    """
    source_data = numpy.array(source_data)
    avg = numpy.mean(source_data)
    std = numpy.std(source_data)
    # standardeviation
    mean = numpy.mean
    std = numpy.std
    arr = source_data
    normalized_data = (arr - mean(arr, axis=0)) / std(arr, axis=0)
    return normalized_data

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except(ValueError):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise(TypeError("window_size size must be a positive odd number"))
    if window_size < order + 2:
        raise(TypeError("window_size is too small for the polynomials order"))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


if __name__ == '__main__':
    timestamps, datasets = load_data()
    calculate_moving_mean()