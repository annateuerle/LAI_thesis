import gdal
import logging

import osr
import gdal
#inDS = gdal.open(r'c:\somedirectory\myRaster.tif')
#inSRS_wkt = inDS.GetProjection()  # gives SRS in WKT
#inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
#inSRS_converter.ImportFromWkt(inSRS)  # populates the spatial ref object with our WKT SRS
#inSRS_forPyProj = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)
log.addHandler(logging.StreamHandler())


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

    geotransform = dataset.GetGeoTransform()
    inSRS = dataset.GetProjection()  # gives SRS in WKT
    inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
    inSRS_converter.ImportFromWkt(inSRS)  # populates the spatial ref object with our WKT SRS
    projection = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
    # print(projection)

    if geotransform:
        log.info("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        log.info("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
        log.info(geotransform)

    log.debug('Raster Count %d', dataset.RasterCount)

    for i, ds in enumerate(dataset.GetSubDatasets()):
        log.debug('%d %s', i+1, ds)

    call_back(dataset, geotransform, projection)


def determine_xy(band, geotransform, projection, lon, lat):
    """
    Given dataset / matrix and geotransform we find
    the nearest x,y close to the given lat lon
    """
    from pyproj import Proj
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

