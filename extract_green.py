"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu.

We extract locations of green area.
"""

import os
import gdal
import glob
import numpy
import settings
from matplotlib import pyplot
import matplotlib as mpl
import logging
import numpy.ma as ma

from read_modis import get_meta_geo_info
from read_modis import determine_lonlat

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

from read_modis import load_modis_data


def extract_green(dataset, geotransform, projection):
    """"Extract green locations from map.
    """

    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    log.debug('Data type %s', bandtype)

    data = band.ReadAsArray()
    raw_data = numpy.copy(data)

    log.debug('Bytes: %s Size %.5d kb', data.nbytes, float(data.nbytes) / 1024)

    passer = numpy.logical_and(data > 0, data <= 1000)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
        data[passer].min(), data[passer].max(),
        data[passer].mean(), data[passer].std())
    )
    cmap = mpl.colors.ListedColormap([
        'gray',
        'lightgreen', 'green', 'green', 'darkgreen',
        'darkgray',
    ])

    norm = mpl.colors.Normalize(vmin=0, vmax=6, clip=True)
    pyplot.imshow(data, norm=norm, cmap=cmap)
    pyplot.colorbar()
    pyplot.show()

    green = ma.masked_inside(data, 1, 5)

    xarr, yarr = numpy.where(green.mask)
    data[green.mask] = 0
    pyplot.imshow(data, norm=norm, cmap=cmap)
    pyplot.colorbar()
    pyplot.show()

    # log.info('Converting xy to lon lat Locations')
    # lons, lats = determine_lonlat(geotransform, projection, xarr[:], yarr[:])
    # log.info('Extracted %d Locations', len(lons))
    return dataset, raw_data, green, xarr, yarr


ground_usage = os.path.join(
    settings.PROJECT_ROOT,
    "Landuse_german",
    "MCD12Q1.A2011001.h18v03.051.2014288191624.hdf"
)


HDF_SOURCES = [
    f'HDF4_EOS:EOS_GRID:"{ground_usage}":MOD12Q1:Land_Cover_Type_5',
]


def extract():
    dataset, geotransform, projection = load_modis_data(HDF_SOURCES[0])
    return extract_green(dataset, geotransform, projection)


def print_hdf_info():
    """Extract hdf layer info

    with extracted layer infromation we can fill configuration.
    HDF_SOURCES

    :return: log information.
    """

    # hdf LAI directory data
    # hdf_files = glob.glob('D:/LAI_thesis/*.hdf')
    hdf_files = glob.glob('D:/LAI_thesis/Landuse_german/*.hdf')

    if not hdf_files:
        raise ValueError('Directory hdf4 source wrong.')

    for hdf_name in hdf_files:
        load_modis_data(hdf_name)


if __name__ == '__main__':
    # print_hdf_info()
    extract()
