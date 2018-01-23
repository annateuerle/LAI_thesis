"""
Parse and load hdf4 modis files, needs gdal 2.1.* on ubuntu
"""
import gdal
import numpy
from matplotlib import pyplot
import logging
filename = 'HDF4_EOS:EOS_GRID:"MOD15A2H.A2017217.h18v03.006.2017234151733.hdf":MOD_Grid_MOD15A2H:Lai_500m'

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


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

    #scale
    passer = numpy.logical_and(lai > 0, lai <= 6)

    log.debug('Min: %5s Max: %5s Mean:%5.2f  STD:%5.2f' % (
                lai[passer].min(), lai[passer].max(),
                lai[passer].mean(), lai[passer].std())
    )

    pyplot.imshow(lai, interpolation='nearest', vmin=0, vmax=250, cmap=pyplot.cm.gist_earth)
    pyplot.colorbar()
    pyplot.show()

    return


if __name__ == '__main__':
    process_modis(filename, process_data)
