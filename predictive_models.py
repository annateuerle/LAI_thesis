import time_series
import logging
import glob
import read_modis
import pyplot
import h5py


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lai_values = []

settings = {
    # square we extract
    'DELTA': 5,
    # longnitude and latitude of the location.
    'LON': 102.500,  # Y
    'LAT': 4.819, # X
    'X': None,
    'Y': None,
}


def do_stuff(dataset, geotransform, projection):
    pass

def do_science():
    # hdf LAI directory data
    #hdf_dir = '/home/stephan/Desktop/data_uu/22978/*/*.hdf'
    hdf_dir = 'D:/LAI_thesis/MODIS_NL_2001_2010/*.hdf'
    hdf_files = glob.glob(hdf_dir)
    #hdf_files.sort()
    if not hdf_files:
        raise ValueError('Directory hdf4 lai source wrong.')

    for hdf_name in hdf_files:
        log.debug('loading %s', hdf_name)
        read_modis.process(
            f'HDF4_EOS:EOS_GRID:"{hdf_name}":MOD_Grid_MOD15A2:Lai_1km',
            do_stuff)

    delta = settings['DELTA']
    surface = delta*delta

    lai_values.sort()

    for x in range(delta*delta):
        x_lai_values = []
        time_x = []
        for m_date, rectangle in lai_values:
            x_lai_values.append(rectangle[x])
            time_x.append(m_date)
            log.error('%s %s', m_date, hdf_name)

        pyplot.plot(time_x, x_lai_values)
        pyplot.title("LAI for 2001-2010")
        pyplot.show()

    pyplot.show()


if __name__ == '__main__':
    do_science()