"""
We prepare data for LAI prediction function calculations.

1. Extract green locations from map.
2. Collect all CRU data for green locations and store in hdf file.
    - normalized?
    - 10 years
    - use geotransform from modis file to make selection in cru data!
3. Collect all LAI data for green locations
    - smoothed.
    - normalized.
    - 10 years.
    - resolution of LAI map is higher.
4. Calculate best lai predictor function for each green location.
    - draw map with colors from this.
    - maybe just a subset of these locations.
5. Improve predictions functions and run step 4 again.
"""

import logging
import os
import extract_green
import h5py
import glob
import plot_map_progress
import settings
from settings import conf
import extract_CRU
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
        size_x = dataset.RasterXSize
        size_y = dataset.RasterYSize

        if x > (size_x + 10):
            continue
        if x < 0:
            continue

        if y > (size_y + 10):
            continue
        if y < 0:
            continue

        points.append((x, y))
    return points


def collect_cru():
    # first extract green map.
    ds, values, green, xarr, yarr = extract_green.extract()

    # open starage hdf5 file.
    storage_name = conf['hdf5storage']
    hdf5 = h5py.File(storage_name, 'a')

    bbox = read_modis.make_lonlat_bbox(ds)
    log.debug('BBOX: %s', bbox)
    bbox_lai = extract_lai_bbox()

    grid_lai = find_xy_cru_grid(ds, bbox_lai)
    # grid = []
    lon_lat_grid, grid_idx = extract_CRU.grid_for(bbox)
    # convert lat,lon to x, y
    grid = find_xy_cru_grid(ds, lon_lat_grid)

    def pred_cru():
        for x, y in zip(xarr, yarr):
            # extract_climatic_variable_from_CRU.extract_for(lon, lat)
            # log.debug('%d %d', x, y)
            yield x, y, 2
            # yield x, y, 10

    hdf5.close()

    plot_map_progress.run_map(
        pred_cru, values, green,
        [
            grid_lai,
            grid
        ],
        modulo=10000)


def extract_lai_bbox():

    source_hdf_dir = os.path.join(
        settings.PROJECT_ROOT,
        settings.lai_locations[conf['groupname']],
    )

    hdf_files = glob.glob(os.path.join(source_hdf_dir, '*.hdf'))

    if not hdf_files:
        log.exception(source_hdf_dir)
        raise ValueError('Directory hdf4 LAI source wrong.')

    for hdf_name in hdf_files[:1]:
        ds, geotransform, projection = read_modis.load_modis_data(
            # hdf_name,
            # f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2H:Lai_500m',
            f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            # f'HDF4_EOS: EOS_GRID:"{hdf_name}": MOD12Q1:Land_Cover_Type_5',
            # f'HDF4_EOS:EOS_GRID:"ground_usage":MOD12Q1:Land_Cover_Type_5',
            )
        bbox = read_modis.make_lonlat_bbox(ds)

    return bbox


if __name__ == '__main__':
    collect_cru()
