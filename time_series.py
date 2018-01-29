"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
"""
import gdal
import glob
import numpy
import math
from matplotlib import pyplot
import matplotlib
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lai_values = []

settings = {
    # square we extract
    'DELTA': 2,
    # somewhere in german forest.
    'LON': 7.5046,  # Y
    'LAT': 51.1156, # X
    'X': None,
    'Y': None,
}


def process_modis(filename, call_back):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    log.info("Driver: {}/{}".format(
        dataset.GetDriver().ShortName,
        dataset.GetDriver().LongName))
    log.info("Size is {} x {} x {}".format(
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount))

    log.info("Projection is {}".format(dataset.GetProjection()))

    geotransform = dataset.GetGeoTransform()

    if geotransform:
        log.info("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        log.info("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
        log.info(geotransform)

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset, geotransform)


def determine_xy(band, geotransform):
    """
    Given dataset / matrix and geotransform we find
    the nearest x,y close to the given lat lon
    """
    from pyproj import Proj
    # +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs
    p_modis_grid = Proj('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')
    x, y = p_modis_grid(settings['LON'], settings['LAT'])
    # or the inverse, from x, y to lon, lat
    #lon, lat = p_modis_grid(x, y, inverse=True)
    log.debug(f'X:{x} Y:{y}')
    # now correct for origin and devide by pixelsize to get x,y in data file.
    pixelsize = 926.625433055833
    return int(x / pixelsize), int((abs(int(y) - 6671703.118)) / pixelsize)


def process_data(dataset, geotransform):

    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    log.debug('Data type %s', bandtype)

    lai = band.ReadAsArray()
    log.debug('Bytes: %s Size %.5d kb', lai.nbytes, float(lai.nbytes) / 1024)

    x, y = determine_xy(band, geotransform)

    log.debug(f'XY: {x}:{y}')

    passer = numpy.logical_and(lai > 0, lai <= 6)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
                lai[passer].min(), lai[passer].max(),
                lai[passer].mean(), lai[passer].std())
    )

    #lai[lai > 7] = 7
    # store specific location in array.

    delta = settings['DELTA']
    cell = []

    for xd in range(delta):
        for yd in range(delta):
            value = lai[y+yd][x+xd]
            cell.append(value)

    assert len(cell) == delta*delta
    lai_values.append(cell)

    # pyplot.imshow(lai, vmin=0, vmax=26)
    # pyplot.plot([x], [y], 'ro')
    # pyplot.colorbar()
    # pyplot.show()

    return


def do_science():
    #hdf LAI directory data
    #C:\Users\DPiA\Downloads\22978
    hdf_dir = 'C://Users/DPiA/Downloads/22978/*/*.hdf'
    hdf_dir = '/home/stephan/Desktop/data_uu/22978/*/*.hdf'
    hdf_files = glob.glob(hdf_dir)
    hdf_files.sort()
    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
        log.debug('loading %s', hdf_name)
        process_modis(
            f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2_927:Lai_1km',
            process_data)

    time_x = range(len(lai_values))
    delta = settings['DELTA']
    surface = delta*delta

    for x in range(delta*delta):
        x_lai_values = []
        for rectangle in lai_values:
            x_lai_values.append(rectangle[x])

        pyplot.plot(time_x, x_lai_values)

    pyplot.show()


if __name__ == '__main__':
    do_science()
