"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
"""
import gdal
import glob
import numpy
from matplotlib import pyplot
import matplotlib
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lai_values = []
lai_values2 = []
lai_values3 = []

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
    lai_values.append(lai[39][442])
    lai_values2.append(lai[42][444])
    lai_values3.append(lai[37][440])

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

    pyplot.plot(time_x, lai_values)
    pyplot.plot(time_x, lai_values2)
    pyplot.plot(time_x, lai_values3)
    pyplot.show()