
# coding: utf-8
import datetime # Python standard library datetime  module
import numpy
import netCDF4
from netCDF4 import Dataset  as netcdf

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

nc = netcdf('cru_ts3.24.01.2001.2010.pre.dat.nc','r')


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
lats = nc.variables['lat'][:]  # extract/copy the data
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
#tmn = nc_fid.variables['tmn'][:]  # shape is time, lat, lon as shown above
pre = nc.variables['pre'][:]


time_idx = 12  # some random month
# Python and the renalaysis are slightly off in time so this fixes that problem
offset = datetime.timedelta(hours=48)
# List of all times in the file as datetime objects

def fix_time():
    dt_time = []
    for t in time:
        start = datetime.date(1900, 1, 1)  # This is the "days since" part

        delta = datetime.timedelta(int(t))  # Create a time delta object from the number of days
        offset = start + delta  # Add the specified number of days to 1900
        dt_time.append(offset)
    cur_time = dt_time[time_idx]
    return cur_time, dt_time


def draw_basemap():
    """Plot of global temperature on our random day"""
    #
    fig = plt.figure()
    #fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
    # for other projections.
    #m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
    m = Basemap(projection='moll', resolution='c', lon_0=0)

    m.drawcoastlines()
    m.drawmapboundary()
    # Make the plot continuous
    pre_cyclic, lons_cyclic = addcyclic(pre[time_idx, :, :], lons)
    # Shift the grid so lons go from -180 to 180 instead of 0 to 360.
    #pre_cyclic, lons_cyclic = shiftgrid(180., pre_cyclic, lons_cyclic, start=False)
    # Create 2D lat/lon arrays for Basemap
    lon2d, lat2d = numpy.meshgrid(lons_cyclic, lats)
    # Transforms lat/lon into plotting coordinates for projection
    x, y = m(lon2d, lat2d)
    # Plot of pre with 11 contour intervals
    cs = m.contourf(x, y, pre_cyclic, 50, cmap=plt.cm.Spectral_r)
    cbar = plt.colorbar(cs, orientation='horizontal', shrink=0.9)
    cbar.set_label("Anna pre plot(ml)")
    global cur_time
    plt.title("%s on %s" % ("Anna pre PLOT", cur_time))
    plt.show()
    fig.show()



def draw_plot():

    darwin = {'name': 'Darwin, Australia', 'lat': -12.45, 'lon': 130.83}

    # Find the nearest latitude and longitude for Darwin
    lat_idx = np.abs(lats - darwin['lat']).argmin()
    lon_idx = np.abs(lons - darwin['lon']).argmin()

    # A plot of the temperature profile for Darwin in 2012
    fig = plt.figure()
    dt_lty = dt_time[-24:]
    pre_lty = pre[-24:]

    plt.plot(dt_lty, pre_lty[:, lat_idx, lon_idx], c='r')
    plt.plot(dt_lty[time_idx], pre_lty[time_idx, lat_idx, lon_idx], c='b', marker='o')
    plt.text(dt_lty[time_idx], pre_lty[time_idx, lat_idx, lon_idx], cur_time, ha='right')

    # fig.autofmt_xdate()
    # plt.ylabel("%s (%s)" % (nc.variables['pre'].var_desc,\
    #                        nc.variables['pre'].units))
    plt.xlabel("Time")
    # plt.title("%s from\n%s for %s" % (nc.variables['pre'].var_desc,\
    #                                  darwin['name'], cur_time.year))
    plt.show()
    fig.show()

cur_time, dt_time = fix_time()
#draw_plot()
draw_basemap()
#print(pre[119][100][:])