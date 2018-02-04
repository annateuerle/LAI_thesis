import time_series
import logging
import glob
import read_modis
from matplotlib import pyplot
import datetime
from datetime import date
import h5py

from settings import settings


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def do_science():
    storage_name = settings['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:

        timestamps = data_file['timestamps']

        time_x = []
        for t in timestamps:
            dt = date.fromtimestamp(t)
            time_x.append(dt)

        y_lai_values = data_file['LAI_german_forest']
        y_lai_values = list(y_lai_values)

    pyplot.plot(time_x, y_lai_values)
    pyplot.title("LAI for 2001-2010 Month")
    pyplot.show()


if __name__ == '__main__':
    do_science()