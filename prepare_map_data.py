"""
We prepare data for LAI prediction function calculaions.

1. Extract green locations from map.
2. Collect all CRU data for green locations and store in hdf file.
    - normalized?
    - 10 years.
    - use geotransform from modis file to make selection in cru data!
3. Collect all LAI data fot green locations
    - smoothed.
    - normalized.
    - 10 years.
    - resolution of LAI map is higer.
4. Calculate best lai predictor function for each green loaction.
    - draw map with colors from this.
    - mabybe just a subset of these locations.
5. Improve predictions functions and run step 4 again.
"""

import logging
import extract_green
import h5py
import plot_map_progress
from settings import settings
import extract_climatic_variable_from_CRU
import read_modis

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

CRU_LOC = {}


def find_xy_cru_grid(dataset, grid):

    geotransform, projection = read_modis.get_meta_geo_info(dataset)
    points = []

    for lon, lat in grid:
        x, y = read_modis.determine_xy(geotransform, projection, lon, lat)
        points.append((x, y))
        #assert x > -1
        #assert y > -1
    return points


def collect_cru():
    # first extract green map.
    ds, dataset, green, xarr, yarr = extract_green.extract()

    # open hdf5 file.
    storage_name = settings['hdf5storage']
    hdf5 = h5py.File(storage_name, 'a')

    bbox = read_modis.make_lonlat_bbox(ds)
    log.debug('BBOX: %s', bbox)
    grid = extract_climatic_variable_from_CRU.extract_for(*bbox, hdf5=hdf5)
    # convert lat,lon to x, y
    grid = find_xy_cru_grid(ds, grid)

    def pred_cru():
        for x, y in zip(xarr, yarr):
            # extract_climatic_variable_from_CRU.extract_for(lon, lat)
            # log.debug('%d %d', x, y)
            yield x, y, 2
            # yield x, y, 10

    hdf5.close()

    plot_map_progress.run_map(pred_cru, dataset, green, grid, modulo=10000)


if __name__ == '__main__':
    collect_cru()
