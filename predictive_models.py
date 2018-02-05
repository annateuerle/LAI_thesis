import time_series
import logging
import glob
import read_modis
from matplotlib import pyplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import datetime
from datetime import date
import h5py
import sympy

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

    time_x = timestamps
    y_lai_values = datasets['lai']
    y_tmp_values = datasets['tmp']
    y_pred_lai = datasets['pred_tmp']


    lai, = pyplot.plot(time_x,  y_lai_values, label='lai')
    pred, = pyplot.plot(time_x, y_pred_lai[:109],  label='pred')
    tmp, = pyplot.plot(time_x,  y_tmp_values[:109], label='tmp')

    pyplot.title("LAI for 2001-2010 Month")
    # Create a legend for the first line.
    #plt.legend(handles=lai, loc=1)
    # Create another legend for the second line.
    #plt.legend(handles=tmp, loc=4)

    plt.legend(handles=[tmp, pred], loc=2)

    pyplot.show()

def make_prediction():
    ds_tmp = datasets['tmp']
    pred_lai = [lai_pred_tmp(tmp) for tmp in ds_tmp]
    datasets['pred_tmp'] = pred_lai

def lai_pred_tmp(tmp):

    return 0.2 * tmp - 2

if __name__ == '__main__':
    load_data()
    make_prediction()
    plot()