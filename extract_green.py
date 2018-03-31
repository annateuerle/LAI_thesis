"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu.

We extract locations of green area.
"""

import gdal
import glob
import numpy
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
    log.debug('Bytes: %s Size %.5d kb', data.nbytes, float(data.nbytes) / 1024)

    passer = numpy.logical_and(data > 0, data <= 1000)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
        data[passer].min(), data[passer].max(),
        data[passer].mean(), data[passer].std())
    )
    cmap = mpl.colors.ListedColormap([
        'gray',
        'lightgreen', 'green', 'green', 'darkgreen',
        'darkgray'
    ])

    norm = mpl.colors.Normalize(vmin=0, vmax=6, clip=True)
    img = pyplot.imshow(data, norm=norm, cmap=cmap)

    pyplot.colorbar()
    pyplot.show()

    green = ma.masked_inside(data, 1, 5)

    return data, green

    xarr, yarr = numpy.where(green.mask)
    data[green.mask] = 0
    pyplot.imshow(data, norm=norm, cmap=cmap)
    pyplot.colorbar()
    pyplot.show()

    #print(len(xarr))
    #print(len(yarr))

    lons, lats = determine_lonlat(geotransform, projection, xarr[:], yarr[:])

    log.info(lons[:10])
    log.info(lats[:10])

    return lons, lats

HDF_SOURCES = [
    # hdf_name,
    'HDF4_EOS:EOS_GRID:"D:/LAI_thesis/Landuse_german\\MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5',
    # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
    # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
    # f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
]

def main():
    dataset, geotransform, projection = load_modis_data(HDF_SOURCES[0])
    extract_green(dataset, geotransform, projection)


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
    main()
