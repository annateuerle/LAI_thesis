"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
and create monthly time-series of lai values which matches the cru data.
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
import load_datasets

from settings import settings
from settings import locations

data_matrix = np.random.uniform(-1, 1, size=(10, 3))

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

LAI_VALUES = []


def process_data(dataset, geotransform, projection, plot_map=False):
    """

    :param dataset: hdf source
    :param geotransform: hdf geotransform information
    :param projection: projection

    adds (datetime, lai_value) to global LAI_VALUES
    """

    band = dataset.GetRasterBand(1)

    metadata = dataset.GetMetadata_Dict()

    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    # log.debug('Data type %s', bandtype)

    lai = band.ReadAsArray()
    # log.debug('Bytes: %s Size %.5d kb', lai.nbytes, float(lai.nbytes) / 1024)
    lat = locations[settings['groupname']]['lat']
    lon = locations[settings['groupname']]['lon']

    x, y = read_modis.determine_xy(geotransform, projection, lon, lat)

    log.debug(f'XY: {x}:{y}')
    log.debug(f'lat:{lat},lon:{lon}')

    # values are 10* real values.
    value = lai[y][x] / 10
    measurement_time = dateparser.parse(metadata['RANGEBEGINNINGDATE'])

    LAI_VALUES.append((measurement_time, value))

    # plot map
    if plot_map:
        pyplot.imshow(lai, vmin=0, vmax=26)
        pyplot.plot([x], [y], 'ro')
        pyplot.colorbar()
        pyplot.show()

    return


def load_lai_from_hdf():
    """
    Given location and hdf direcotry in settings load all LAI
    values for given location over time.
    Each hdf file is a 8 day average.
    :return:  global LAI_VALUES will be filled.
    """
    # hdf LAI directory data
    #hdf_dir = '/home/stephan/Desktop/data_uu/22978/*/*.hdf'

    hdf_dir = settings['hdf_dir']
    hdf_files = glob.glob(hdf_dir)

    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    hdf_files.sort()

    for hdf_name in hdf_files:
        log.debug('loading %s', hdf_name)
        read_modis.process(
            f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            process_data)

    # sort values by date.
    LAI_VALUES.sort()

    x_lai_values = []
    time_x = []

    for m_date, value in LAI_VALUES:
        x_lai_values.append(value)
        time_x.append(m_date)

    pyplot.plot(time_x, x_lai_values)
    pyplot.title("LAI for 2001-2010  HDF 8-day")
    pyplot.show()


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
            lai_for_every_day.append((oneday, lai_value))

    return lai_for_every_day


def store_avg_month(lai_values_months, month_values, current_month):

    if month_values:
        avg_month = float(sum(month_values)) / len(month_values)
        lai_values_months.append((current_month, avg_month))
        log.debug("Stored Month: %s Value %f", current_month, avg_month)


def convert_to_120months(hdf_lai_values: list, smooth=True) -> list:
    """"
    Convert 8 day period to month period.
    """
    lai_values_months = []
    # make sure we are sorted by date
    hdf_lai_values.sort()

    # smooth out LAI values. using savitzky_golay
    if smooth:
        t_values = [d for (d, v) in hdf_lai_values]
        l_values = [v for (d, v) in hdf_lai_values]
        smooth_l = load_datasets.savitzky_golay(l_values, 9, 4)
        smooth_lai = list(zip(t_values, smooth_l))
    else:
        smooth_lai = hdf_lai_values

    log.error(f'smooth: {smooth}')
    log.error(list(smooth_lai))

    lai_for_every_day = create_lai_for_every_day(smooth_lai)
    assert lai_for_every_day

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

    assert lai_values_months
    return lai_values_months


def plot_month(lai_values_by_month, smooth=[]):
    """
    :param lai_values_by_month:
    :param smooth: also plot smoothed version
    :return: PLOT of lai values.
    """

    y_lai_values = []
    time_x = []
    s_lai = []   # smoothed lai series.

    if smooth:
        assert len(smooth) == len(lai_values_by_month)

    for i, (m_date, lai) in enumerate(lai_values_by_month):
        y_lai_values.append(lai)
        time_x.append(m_date)
        if smooth:
            s_lai.append(smooth[i][1])

    if smooth:
        assert len(y_lai_values) == len(s_lai)

    pyplot.plot(time_x, y_lai_values, label='Original LAI dataset' )

    if smooth:
        pyplot.plot(time_x, s_lai, label='smooth')

    pyplot.title(f"LAI for 2001-2010 Month")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    load_lai_from_hdf()
    lai_by_month = convert_to_120months(LAI_VALUES, smooth=False)
    lai_by_month_smooth = convert_to_120months(LAI_VALUES)

    for month, value in lai_by_month:
        log.debug('%s %f', month, value)

    for month, value in lai_by_month_smooth:
        log.debug('%s %f', month, value)

    plot_month(lai_by_month, smooth=lai_by_month_smooth)
    # save the lai data.
    save_lai_location(lai_by_month_smooth)
