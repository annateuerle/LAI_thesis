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
import plot_map_progress
import numpy
import random

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

CRU_LOC = {}

def collect_cru():
    # first extract green map.
    data, green, lons, lats, xarr, yarr, geotransform = extract_green.extract()

    # open hdf5 file.
    storage_name = settings['hdf5storage']
    hdf5 = h5py.File(storage_name, 'w')

    def load_cru():
        for lon, lat, x, y in zip(lons, lats, xarr, yarr):
            # extract_climatic_variable_from_CRU.extract_for(geotransform, hdf5=hdf5)
            #extract_climatic_variable_from_CRU.extract_for(lon, lat)
            # log.debug('%d %d', x, y)
            yield x, y, 2
            # yield x, y, 10

    plot_map_progress.run_map(load_cru, data, green,  modulo=1000)
    hdf5.close()


if __name__ == '__main__':
    collect_cru()