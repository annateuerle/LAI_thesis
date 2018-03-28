#Plot of the CRU dataset as a function of time 2001-2010 and a global map of some day from that period.

import datetime # Python standard library datetime  module
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
nc_var = settings['ncvar']

nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{nc_var}.dat.nc','r')



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



nc_attrs, nc_dims, nc_vars = ncdump(nc)
# Extract data from NetCDF file
lats = nc.variables['lat'][:]   # extract/copy the data
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
nc_ds = nc.variables[nc_var][:]

# Find the nearest latitude and longitude
lat_idx = numpy.abs(lats - locations[settings['groupname']]['lat']).argmin()
lon_idx = numpy.abs(lons - locations[settings['groupname']]['lon']).argmin()

settings['X'] = lat_idx
settings['Y'] = lon_idx

def fix_time():
    """
    # List of all times in the file as datetime objects
    """
    dt_time = []
    for t in time:
        start = datetime.date(1900, 1, 1)  # This is the "days since" part
        delta = datetime.timedelta(int(t))  # Create a time delta object from the number of days
        offset = start + delta # Add the specified number of days to 1900
        dt_time.append(offset)
        # print(offset)

    return dt_time


def draw_basemap():
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


def draw_plot(time_idx):
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


def save_lai_location():
    # Write CRU data to HDF5

    values_at_loc = nc_ds[:, settings['X'], settings['Y']]

    print(len(dt_time))

    storage_name = settings['hdf5storage']
    data_file = h5py.File(storage_name, 'a')
    nc_matrix = np.array(
        values_at_loc
    )
    print(nc_matrix)

    groupname = f"{settings['groupname']}-{settings['ncvar']}"

    if groupname in data_file.keys():
        del data_file[groupname]
        log.debug('deleted %s', groupname)
    data_file.create_dataset(groupname, data=nc_matrix)
    data_file.close()

    log.debug(f'Saved CRU {groupname}')


dt_time = fix_time()

draw_plot(settings['time_idx'])
draw_basemap()
save_lai_location()
