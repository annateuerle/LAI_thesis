"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
"""
import gdal
import glob
import dateparser
import math
from matplotlib import pyplot
import matplotlib
import logging
import read_modis
import h5py
import numpy as np
import datetime

from settings import settings

data_matrix = np.random.uniform(-1, 1, size=(10, 3))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lai_values = []


def process_data(dataset, geotransform, projection):

    band = dataset.GetRasterBand(1)

    metadata = dataset.GetMetadata_Dict()

    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    # log.debug('Data type %s', bandtype)

    lai = band.ReadAsArray()
    # log.debug('Bytes: %s Size %.5d kb', lai.nbytes, float(lai.nbytes) / 1024)

    x, y = read_modis.determine_xy(
        band, geotransform, projection, settings['LON'], settings['LAT'])

    #log.debug(f'XY: {x}:{y}')

    #passer = numpy.logical_and(lai > 0, lai <= 6)

    #log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
    #            lai[passer].min(), lai[passer].max(),
    #            lai[passer].mean(), lai[passer].std())
    #)

    #lai[lai > 7] = 7
    # store specific location in array.

    delta = settings['DELTA']
    cell = []

    for xd in range(delta):
        for yd in range(delta):
            value = lai[y+yd][x+xd] / 10
            cell.append(value)

    assert len(cell) == delta*delta
    measurement_time = dateparser.parse(metadata['RANGEBEGINNINGDATE'])
    lai_values.append((measurement_time, cell))

    #pyplot.imshow(lai, vmin=0, vmax=26)
    #pyplot.plot([x], [y], 'ro')
    #pyplot.colorbar()
    #pyplot.show()

    return


def do_science():
    # hdf LAI directory data
    #hdf_dir = '/home/stephan/Desktop/data_uu/22978/*/*.hdf'

    hdf_dir = settings['hdf_dir']
    hdf_files = glob.glob(hdf_dir)
    # hdf_files.sort()
    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
        log.debug('loading %s', hdf_name)
        read_modis.process(
            f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            process_data)

    delta = settings['DELTA']
    surface = delta*delta

    lai_values.sort()

    for x in range(delta*delta):
        x_lai_values = []
        time_x = []
        for m_date, rectangle in lai_values:
            x_lai_values.append(rectangle[x])
            time_x.append(m_date)
            log.error('%s %s', m_date, hdf_name)

    #pyplot.plot(time_x, x_lai_values)
    #pyplot.title("LAI for 2001-2010")
    #pyplot.show()


def save_lai_location(lai_array):
    """ Write data to HDF5
    """

    storage_name = settings['hdf5storage']

    lai_matrix = np.array(
        [cell for time, cell in lai_array]
    )

    time_matrix = [time.timestamp() for time, cell in lai_array]

    log.debug('X- time count %d', len(time_matrix))

    groupname = settings['groupname'] + '-lai'

    def set_dataset(hdf, groupname, data):
        """replace of set data in groupname of hdf file"""
        if groupname in hdf.keys():
            del hdf[groupname]
        hdf.create_dataset(groupname, data=data)

    with h5py.File(storage_name, "a") as data_file:
        set_dataset(data_file, groupname, lai_matrix)
        set_dataset(data_file, 'timestamps', time_matrix)
        # del data_file['LAI_german_forest_monthX']

    log.debug(f'Saved LAI {groupname}')


def create_lai_for_every_day(hdf_lai_values):
    """create day values for all days."""
    lai_for_every_day = []

    for d8t, lai_value in hdf_lai_values:
        for i in range(8):
            day1 = datetime.timedelta(days=1)
            oneday = d8t + i * day1
            lai_for_every_day.append((oneday, lai_value[0]))

    return lai_for_every_day


def store_avg_month(lai_values_months, month_values, current_month):

    if month_values:
        avg_month = float(sum(month_values)) / len(month_values)
        lai_values_months.append((current_month, avg_month))
        log.debug("Stored Month: %s Value %f", current_month, avg_month)

def convert_to_120months(hdf_lai_values: list) -> list:
    ""
    """"
    Convert 8 day period to month period.
    
    We asume the hdf_lai_values are sorted by date.
    """
    lai_values_months = []

    lai_for_every_day = create_lai_for_every_day(hdf_lai_values)

    current_month = 0
    month_values = []
    year = 0
    current_month_date = None

    # create month avg values of lai
    for day, day_value in lai_for_every_day:
        month = day.month

        if current_month is None:
            current_month = month
            year = day.year
            current_month_date = datetime.datetime(year=day.year, month=day.month, day=15)

        if month is not current_month:
            store_avg_month(lai_values_months, month_values, current_month_date)
            # reset month counting.
            current_month = month
            month_values = []
            year = day.year
            current_month_date = datetime.datetime(year=year, month=month, day=15)

        # store value of one day.
        month_values.append(day_value)

    if month_values:
        # store last month
        current_month_date = datetime.datetime(year=year, month=current_month, day=15)
        store_avg_month(lai_values_months, month_values, current_month_date)
        log.debug(len(lai_values_months))

    return lai_values_months

def plot_month(lai_values_by_month):

    x_lai_values = []
    time_x = []

    for m_date, rectangle in lai_values_by_month:
        x_lai_values.append(rectangle)
        time_x.append(m_date)
        # log.error('%s %s', m_date, hdf_name)

    pyplot.plot(time_x, x_lai_values)
    pyplot.title("LAI for 2001-2010 Month")
    pyplot.show()


if __name__ == '__main__':
    do_science()
    lai_by_month = convert_to_120months(lai_values)

    for month, value in lai_by_month:
        log.debug('%s %f', month, value)

    plot_month(lai_by_month)
    save_lai_location(lai_by_month)
