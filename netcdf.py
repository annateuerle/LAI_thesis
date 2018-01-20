
# coding: utf-8

# In[4]:


import datetime as dt  # Python standard library datetime  module
import numpy as np
import  netCDF4
from netCDF4 import Dataset  as netcdf
nc = netcdf('cru_ts3.24.01.2001.2010.pre.dat.nc','r')
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid


# In[9]:


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


# In[18]:



nc_attrs, nc_dims, nc_vars = ncdump(nc)
# Extract data from NetCDF file
lats = nc.variables['lat'][:]  # extract/copy the data
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
#tmn = nc_fid.variables['tmn'][:]  # shape is time, lat, lon as shown above
pre = nc.variables['pre'][:]


# In[11]:


time_idx = 60  # some random month
# Python and the renalaysis are slightly off in time so this fixes that problem
offset = dt.timedelta(hours=48)
# List of all times in the file as datetime objects
dt_time = [dt.date(1, 1, 1) + dt.timedelta(hours=t) - offset           for t in time]
cur_time = dt_time[time_idx]


# In[ ]:


# Plot of global temperature on our random day
fig = plt.figure()
fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
# Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
# for other projections.
m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90,            llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
m.drawcoastlines()
m.drawmapboundary()
# Make the plot continuous


# In[ ]:


# Make the plot continuous
air_cyclic, lons_cyclic = addcyclic(air[time_idx, :, :], lons)
# Shift the grid so lons go from -180 to 180 instead of 0 to 360.
air_cyclic, lons_cyclic = shiftgrid(180., air_cyclic, lons_cyclic, start=False)
# Create 2D lat/lon arrays for Basemap
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
# Transforms lat/lon into plotting coordinates for projection
x, y = m(lon2d, lat2d)
# Plot of air temperature with 11 contour intervals
cs = m.contourf(x, y, air_cyclic, 11, cmap=plt.cm.Spectral_r)
cbar = plt.colorbar(cs, orientation='horizontal', shrink=0.5)
cbar.set_label("%s (%s)" % (nc_fid.variables['air'].var_desc,                            nc_fid.variables['air'].units))
plt.title("%s on %s" % (nc_fid.variables['air'].var_desc, cur_time))


# In[24]:


darwin = {'name': 'Darwin, Australia', 'lat': -12.45, 'lon': 130.83}

# Find the nearest latitude and longitude for Darwin
lat_idx = np.abs(lats - darwin['lat']).argmin()
lon_idx = np.abs(lons - darwin['lon']).argmin()

# A plot of the temperature profile for Darwin in 2012
#fig = plt.figure()
plt.plot(dt_time, pre[:, lat_idx, lon_idx], c='r')
plt.plot(dt_time[time_idx], pre[time_idx, lat_idx, lon_idx], c='b', marker='o')
plt.text(dt_time[time_idx], pre[time_idx, lat_idx, lon_idx], cur_time,         ha='right')
#fig.autofmt_xdate()
#plt.ylabel("%s (%s)" % (nc.variables['pre'].var_desc,\
#                        nc.variables['pre'].units))
plt.xlabel("Time")
#plt.title("%s from\n%s for %s" % (nc.variables['pre'].var_desc,\
#                                  darwin['name'], cur_time.year))
plt.show()

