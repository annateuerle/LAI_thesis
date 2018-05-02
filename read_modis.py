""""
Read MODIS datasets.
"""

import logging
import numpy as np
import math

import osr
import gdal
from pyproj import Proj

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def get_meta_geo_info(dataset):
    geotransform = dataset.GetGeoTransform()
    # gives SRS in WKT
    in_srs = dataset.GetProjection()
    # makes an empty spatial ref object
    in_srs_converter = osr.SpatialReference()
    # populates the spatial ref object with our WKT SRS
    in_srs_converter.ImportFromWkt(in_srs)
    # Exports an SRS ref as a Proj4 string usable by PyProj
    projection = in_srs_converter.ExportToProj4()
    return geotransform, projection


def determine_xy(geotransform, projection, lon, lat):
    """Calculate x, y in dataset.

    Given geotransform , projection, latitude and longitude
    we find the nearest x, y close to the given lat lon
    """
    # +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs
    p_modis_grid = Proj(projection)
    x, y = p_modis_grid(lon, lat)
    log.debug(f'X:{x} Y:{y}')
    return coord2pixel(geotransform, x, y)


def pixel2coord(geotransform, x, y):
    xoff, a, b, yoff, d, e = geotransform
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return xp, yp


def coord2pixel(geotransform, xp, yp):
    xoff, a, b, yoff, d, e = geotransform
    a1 = np.array([[a, b], [d, e]])
    b1 = np.array([xp-xoff, yp-yoff])
    xy = np.linalg.solve(a1, b1)
    x = round(xy[0])
    y = round(xy[1])
    return [x, y]


def determine_lonlat(geotransform, projection, px, py):
    """
    find lon, lat for given pixel x, y.

    :param geotransform:
    :param projection:
    :param px:
    :param py:
    :return: lon, lat
    """
    x, y = pixel2coord(geotransform, px, py)
    lon, lat = projection(x, y, inverse=True)
    return lon, lat


def make_lonlat_bbox(dataset):
    """Return bbox of lon1, lat1, lon2, lat2
    """

    log.info("Size is {} x {} x {}".format(
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount))

    log.info("Projection is {}".format(dataset.GetProjection()))

    geotransform, projection = get_meta_geo_info(dataset)

    size_x = dataset.RasterXSize
    size_y = dataset.RasterYSize

    proj = Proj(projection)
    lon1, lat1 = determine_lonlat(geotransform, proj, 0, 0)
    lon2, lat2 = determine_lonlat(geotransform, proj, 0, size_y)
    lon3, lat3 = determine_lonlat(geotransform, proj, size_x, 0)
    lon4, lat4 = determine_lonlat(geotransform, proj, size_x, size_y)

    return (
        (lon1, lat1),
        (lon2, lat2),
        (lon3, lat3),
        (lon4, lat4),
    )


def test_location_logic(_dataset, geotransform, projection):
    """For lat, lon covered in dataset test conversion.

    We test that lat lon to x,y and x,y to lat, lon
    yields the same result!

    :param _dataset: not used
    :param geotransform:
    :param projection:
    :return: log output.
    """
    # +proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs    # noqa
    # p_modis_grid = Proj(projection)
    lon = 7.495085457117231
    lat = 51.08333332874323

    log.info('LON %s LAT %s', lon, lat)
    x, y = determine_xy(geotransform, projection, lon, lat)
    log.info('X %s Y %s', x, y)

    p_modis_grid = Proj(projection)
    lon2, lat2 = determine_lonlat(geotransform, p_modis_grid, x, y)
    log.info('LON2 %s LAT2 %s', lon2, lat2)

    log.info(lon2 - lon)
    log.info(lat2 - lat)

    # assert abs(lon2 - lon) == 0.0
    # assert abs(lat2 - lat) == 0.0


def load_modis_data(filename):
    """Extract modis data and META data from modis file.

    we extract
        - raster data
        - geotransform (geo metadata, origin, pixelsize)
        - projection used.

    :param filename: string (gdal) path to hdf
    :return: dataset, geotranform, projection
    """
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
        log.info("Pixel Size = ({}, {})".format(
            geotransform[1], geotransform[5]))

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i + 1, ds)

    return dataset, geotransform, projection


def main():
    filename = 'HDF4_EOS:EOS_GRID:"D:/LAI_thesis/Landuse_german\\MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5'  # noqa
    # filename = 'HDF4_EOS:EOS_GRID:"/media/stephan/blender1/laithesis/Landuse_german/MCD12Q1.A2011001.h18v03.051.2014288191624.hdf":MOD12Q1:Land_Cover_Type_5'  # noqa
    dataset, geotransform, projection = load_modis_data(filename)
    test_location_logic(dataset, geotransform, projection)


if __name__ == '__main__':
    main()
