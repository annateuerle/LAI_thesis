#Plot of the CRU dataset as a function of time 2001-2010 and a global map of some day from that period.

import datetime
import numpy
import logging
import netCDF4
from netCDF4 import Dataset as netcdf

import h5py
import numpy as np

from  matplotlib import pyplot
import mpl_toolkits
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

from settings import settings
from settings import locations

startyear = settings['startyear']
endyear = settings['endyear']
# nc_var = settings['ncvar']

CACHE = {
    'tmp': {},
    'pre': {},
    'var': {},
    'vap': {},
    'pet': {},
}


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def fill_cache(ds_var):
    nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{ds_var}.dat.nc', 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc)
    # Extract data from NetCDF file
    lats = nc.variables['lat'][:]  # extract/copy the data
    lons = nc.variables['lon'][:]
    time = nc.variables['time'][:]
    nc_ds = nc.variables[ds_var][:]

    CACHE[ds_var]['lats'] = lats
    CACHE[ds_var]['lons'] = lons
    CACHE[ds_var]['time'] = fix_time(time)
    CACHE[ds_var]['ds'] = nc_ds


def extract_for(geotransform, hdf5=None):
    """
    Extract for all cru variables all data at locations and
    saves it in hdf5 dataset.

    groupname/lon:lat/nc_var

    :param geotransform:
    :param hdf5: optional hdf5 file.
    :return: None.
    """

    for ds_var in ['tmp', 'vap', 'pet', 'pre']:

        # todo use geotransform.

        # Find the nearest latitude and longitude
        lat_idx = numpy.abs(lats - lat).argmin()  # x.
        lon_idx = numpy.abs(lons - lon).argmin()  # y.

        # used as cache key.
        lon_c = lons[lon_idx]  # x
        lat_c = lats[lat_idx]  # y

        save_location(lat_c, lon_c, lon_idx, lat_idx, ds_var, ds, hdf5=hdf5)


def fix_time(times):
    """
    # List of all times in the file as datetime objects
    """
    dt_time = []
    for t in times:
        start = datetime.date(1900, 1, 1)   # This is the "days since" part
        delta = datetime.timedelta(int(t))  # Create a time delta object from the number of days
        offset = start + delta   # Add the specified number of days to 1900
        dt_time.append(offset)

    return dt_time


def draw_basemap(nc_ds, dt_time, lons, lats):
    """Plot of global temperature on our random day"""
    #
    fig = pyplot.figure()

    #fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
    # for other projections.
    # m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
    m = Basemap(projection='cyl', resolution='c', lon_0=0)

    m.drawcoastlines()
    m.drawmapboundary()
    time_idx = settings['time_idx']
    # Make the plot continuous
    ds_cyclic, lons_cyclic = addcyclic(nc_ds[time_idx, :, :], lons)
    # Shift the grid so lons go from -180 to 180 instead of 0 to 360.
    pre_cyclic, lons_cyclic = shiftgrid(180., ds_cyclic, lons_cyclic, start=False)
    # Create 2D lat/lon arrays for Basemap
    lon2d, lat2d = numpy.meshgrid(lons_cyclic, lats)
    # Transforms lat/lon into plotting coordinates for projection
    x, y = m(lon2d, lat2d)
    # Plot of pre with 11 contour intervals
    cs = m.contourf(x, y, ds_cyclic, 20, cmap=pyplot.cm.Spectral_r)
    cbar = pyplot.colorbar(cs, orientation='horizontal', shrink=0.9)
    dot_time = dt_time[time_idx]
    # cbar.set_label
    pyplot.title(f"Global {nc_var} for {dot_time.year}.{dot_time.month}")
    pyplot.show()


def draw_plot(dt_time, time_idx, lat_idx, lon_idx, nc_ds):
    """
    :param fig:
    :return:
    """

    # A plot
    # fig = pyplot.figure()
    fig = pyplot.figure()
    dot_time = dt_time[time_idx]

    pyplot.plot(dt_time, nc_ds[:, lat_idx, lon_idx], c='r')
    pyplot.plot(dt_time[time_idx], nc_ds[time_idx, lat_idx, lon_idx], c='b', marker='o')
    pyplot.text(dt_time[time_idx], nc_ds[time_idx, lat_idx, lon_idx], dot_time, ha='right')

    # fig.autofmt_xdate()
    # pyplot.ylabel("%s (%s)" % (nc.variables['pre'].var_desc,\
    #                        nc.variables['pre'].units))
    pyplot.xlabel("Time")
    pyplot.title(f"Local {nc_var} from {startyear} to {endyear}")
    pyplot.show()


SEEN = set()

def save_location(lat_c, lon_c, x, y, ds_var, ds, hdf5 = None):
    """
    :param lat: lat used by lai
    :param lon: lon used by lai
    :param lat_c: lat used in cru
    :param lon_c: lon used in cru
    :param x:  lon index
    :param y:  lat index
    :param ds: the nc dataset with x year data
    :param ds_var: nc label of dataset
    :return: Nothing
    """
    if not hdf5:
        storage_name = settings['hdf5storage']
        data_file = h5py.File(storage_name, 'w')
    else:
        data_file = hdf5

    cru_groupname = f"{settings['groupname']}/{lon_c}:{lat_c}/{ds_var}"

    if cru_groupname in SEEN:
        return
        #del data_file[groupname]
        #log.debug('deleted %s', groupname)
    SEEN.add(cru_groupname)
    # Write CRU data to HDF5
    values_at_loc = ds[:, y, x]
    nc_matrix = np.array(
        values_at_loc
    )
    h5ds = data_file.create_dataset(cru_groupname, data=nc_matrix)
    # store meta.
    h5ds.attrs['cru_loc'] = [lon_c, lat_c]
    h5ds.attrs['cru_idx'] = [x, y]

    log.debug(f'Saved CRU {cru_groupname}')

    if not hdf5:
        data_file.close()


if __name__ == '__main__':
    pass
    # dt_time = fix_time()

    # draw_plot(settings['time_idx'])
    # draw_basemap()
    # save_location()