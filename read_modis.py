""""
Read modis datasets.
"""

import gdal
import logging

import osr
import gdal
from pyproj import Proj

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def get_meta_geo_info(dataset):
    geotransform = dataset.GetGeoTransform()
    inSRS = dataset.GetProjection()  # gives SRS in WKT
    inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
    inSRS_converter.ImportFromWkt(inSRS)  # populates the spatial ref object with our WKT SRS
    projection = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
    return geotransform, projection

def process(filename, call_back):
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
        log.info(geotransform)

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset, geotransform, projection)


def determine_xy(geotransform, projection, lon, lat):
    """
    Given dataset / matrix and geotransform we find
    the nearest x,y close to the given lat lon
    """

    # +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs
    p_modis_grid = Proj(projection)
    x, y = p_modis_grid(lon, lat)
    # or the inverse, from x, y to lon, lat
    #lon, lat = p_modis_grid(x, y, inverse=True)
    log.debug(f'X:{x} Y:{y}')
    # now correct for origin and devide by pixelsize to get x,y in data file.
    pixelsize_x, pixelsize_y = geotransform[1], geotransform[5]
    origin_x, origin_y = geotransform[0], geotransform[3]

    x = int(abs(abs(int(x)) - abs(origin_x)) / abs(pixelsize_x))
    y = int(abs(abs(int(y)) - abs(origin_y)) / abs(pixelsize_y))

    return (x, y)


def determine_lonlat(geotransform, projection, xarr, yarr):
    """
    find lon, lat for given x, y.

    :param geostransform:
    :param projection:
    :param x:
    :param y:
    :return: lon, lat
    """
    # now correct for origin and devide by pixelsize to get x,y on globe.
    pixelsize_x, pixelsize_y = geotransform[1], geotransform[5]
    origin_x, origin_y = geotransform[0], geotransform[3]

    abs_x = [origin_x + x * pixelsize_x for x in xarr]
    abs_y = [origin_y + y * pixelsize_y for y in yarr]

    p_modis_grid = Proj(projection)
    lons, lats = p_modis_grid(abs_x, abs_y, inverse=True)
    return lons, lats


def test_location_logic(dataset, geotransform, projection):
    # +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs
    p_modis_grid = Proj(projection)
    lat = 51.1156
    lon = 7.5046
    log.info('LON %s LAT %s', lon, lat)
    x, y = determine_xy(geotransform, projection, lon, lat)
    log.info('X %s Y %s', x, y)
    lon2, lat2 = determine_lonlat(geotransform, projection, [x], [y])
    log.info('LON2 %s LAT2 %s', lon2, lat2)


if __name__ == '__main__':
    process(
        'HDF4_EOS:EOS_GRID:"D:/LAI_thesis/Landuse_german\\MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5',
        test_location_logic
    )