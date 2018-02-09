import time_series
import logging
import glob
import read_modis
import matplotlib.patches as mpatches
from matplotlib import pyplot
import datetime
from datetime import date
import h5py
import sympy
from math import sqrt
import numpy as np



from settings import settings


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

timsetamps = None

datasets = {
    'lai': None,
    'tmp': None,
    'pre': None,
    'vap': None,
    'pet': None,
}


def load_data():
    storage_name = settings['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:

        x_time = data_file['timestamps']

        time_x = []
        for t in x_time:
            dt = date.fromtimestamp(t)
            time_x.append(dt)

        global timestamps
        timestamps = time_x
        groupname = settings['groupname']
        for var in datasets.keys():
            datasets[var] = list(data_file[f'{groupname}-{var}'])

def plot():

    time_x = timestamps[:120]
    y_lai_values = datasets['lai']
    y_tmp_values = datasets['tmp']
    y_pred_lai = datasets['pred_tmp']

    lai, = pyplot.plot(time_x,  y_lai_values[:120], label='lai')
    pred,  = pyplot.plot(time_x, y_pred_lai[:120],  label='pred')
    #tmp, = pyplot.plot(time_x,  y_tmp_values[:109], label='tmp')

    rmse = calc_rmse(datasets['pred_tmp'][:120], datasets['lai'][:120])
    #print('RMSE is:', rmse)

    pyplot.figtext(
        0.81, 0.84, f'rmse {rmse:.3f}', fontsize=10, horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='grey', alpha=0.5),
    )

    #fig = pyplot.figure()
    pyplot.title("LAI for 2001-2010 Month")
    #textplot = fig.add_subplot(111)
    pyplot.legend(handles=[pred, lai], loc=2)

    pyplot.show()

def make_prediction():
    ds_tmp = datasets['tmp']
    pred_lai = [lai_pred_tmp(tmp) for tmp in ds_tmp]
    datasets['pred_tmp'] = pred_lai

def lai_pred_tmp(tmp):

    return 0.1 * tmp + 0.5

def calc_rmse(predictions, targets):

    if type(predictions) == list:
        predictions = np.array(predictions)
        targets = np.array(targets)

    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val  #get the ^

if __name__ == '__main__':
    load_data()
    make_prediction()
    plot()
