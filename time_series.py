"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
"""
import gdal
import glob
import numpy
import struct
from matplotlib import pyplot
import matplotlib
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lai_values = []

settings = {
    # square we extract
    'DELTA': 5
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

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset)


def extract(lat, lon, band, geotransform):
    """
    Given dataset / matrix and geotransform we find
    the nearest x,y close to the given lat lon and
    return value's found.
    """
    # geotransform parameters
    # top left x [0], w-e pixel resolution [1], rotation [2], top left y [3], rotation [4], n-s pixel resolution [5]
    X = geotransform[0]  # top left x
    Y = geotransform[3]  # top left y

    for y in range(band.YSize):

        scanline = band.ReadRaster(
            0,             y,
                                   band.XSize,
                                   1,
                                   band.XSize,
                                   1,
                                   band.DataType)

        values = struct.unpack(fmttypes[BandType] * band.XSize, scanline)

        for value in values:

            if (value == 256):
                print
                "%.4f %.4f %.2f" % (X, Y, value)
            X += geotransform[1]  # x pixel size
        X = geotransform[0]
        Y += geotransform[5]  # y pixel size

    dataset = None


def process_data(dataset):

    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError('Could not read hdf file')

    bandtype = gdal.GetDataTypeName(band.DataType)

    log.debug('Data type %s', bandtype)

    lai = band.ReadAsArray()
    log.debug('Bytes: %s Size %.5d kb', lai.nbytes, float(lai.nbytes) / 1024)

    passer = numpy.logical_and(lai > 0, lai <= 6)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
                lai[passer].min(), lai[passer].max(),
                lai[passer].mean(), lai[passer].std())
    )

    # lai[lai > 7] = 7
    # store specific location in array.

    x = 39
    y = 442
    delta = settings['DELTA']
    cell = []

    for xd in range(delta):
        for yd in range(delta):
            value = lai[x+xd][y+yd]
            cell.append(value)

    assert len(cell) == delta*delta
    lai_values.append(cell)

    #pyplot.imshow(lai, vmin=0, vmax=26)
    #pyplot.colorbar()
    #pyplot.show()

    return


if __name__ == '__main__':
    #hdf LAI directory data
    #C:\Users\DPiA\Downloads\22978
    hdf_files = glob.glob('C://Users/DPiA/Downloads/22978/*/*.hdf')
    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
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