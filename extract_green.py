#Map of the dataset MODIS15A.
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

    geotransform, projection = get_meta_geo_info(dataset)

    if geotransform:
        log.info("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        log.info("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset, geotransform, projection)


def extract_green(dataset, geotransform, projection):

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

    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    img = pyplot.imshow(data, norm=norm, cmap=cmap)

    pyplot.colorbar()
    pyplot.show()
    zeros = numpy.zeros_like(data)
    green = ma.masked_inside(data, 1, 5)
    xarr, yarr = numpy.where(green.mask)
    #print(xarr)
    data[green.mask] = 0
    pyplot.imshow(data, norm=norm, cmap=cmap)
    pyplot.colorbar()
    pyplot.show()

    print(len(xarr))
    print(len(yarr))

    lons, lats = determine_lonlat(geotransform, projection, xarr[:], yarr[:])

    log.info(lons[:10])
    log.info(lats[:10])

    return lons, lats


if __name__ == '__main__':
    #hdf LAI directory data

    #hdf_files = glob.glob('D:/LAI_thesis/*.hdf')
    hdf_files = glob.glob('D:/LAI_thesis/Landuse_german/*.hdf')
    # Landuse.

    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
        process_modis(
            #hdf_name,
            #f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
            #f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            #f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
            'HDF4_EOS:EOS_GRID:"D:/LAI_thesis/Landuse_german\\MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5',
            extract_green)