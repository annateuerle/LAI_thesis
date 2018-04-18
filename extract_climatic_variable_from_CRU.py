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

from settings import conf
from settings import locations

startyear = conf['startyear']
endyear = conf['endyear']
# nc_var = conf['ncvar']

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
    ts_dt = fix_time(time)
    CACHE[ds_var]['time'] = ts_dt
    CACHE[ds_var]['days'] = time
    CACHE[ds_var]['ds'] = nc_ds

    # draw_basemap(nc_ds, ts_dt, lons, lats, ds_var)

def store_grid_from_cru(grid, grid_idx, hdf5=None):
    """Given we know which locations we need to calculated

    extract information in grid and store it in hdf5

    groupname/lon:lat/nc_var
    """

    for ds_var in ['tmp', 'vap', 'pet', 'pre']:
        if not CACHE.get(ds_var):
            fill_cache(ds_var)

        ds = CACHE[ds_var]['ds']

        for i, (lon, lat) in enumerate(grid):
            lon_idx, lat_idx = grid_idx[i]

            values_at_loc = ds[:, lat_idx, lon_idx]
            save_location(lon, lat, ds_var, values_at_loc, hdf5)


def extract_grid_for(bbox) # , hdf5=None, save=True):
    """
    Extract for all cru variables all data at locations and
    saves it in hdf5 dataset.


    :param hdf5: optional hdf5 file.
    :param save: save / update hdf5 data
    :return: None.
    """
    lats = [p[0] for p in bbox]
    lons = [p[1] for p in bbox]

    lat_min = min(lats)
    lon_min = min(lons)

    nc = netcdf(f'cru_ts3.24.01.{startyear}.{endyear}.{ds_var}.dat.nc', 'r')
    nc_attrs, nc_dims, nc_vars = ncdump(nc)
    lats = nc.variables['lat'][:]  # extract/copy the data
    lons = nc.variables['lon'][:]

    lats = numpy.array(lats)
    lons = numpy.array(lons)

    # invalidate all locations outside our given bbox
    # with 1 extra box spacing
    padding_lat = 0.55
    padding_lon = 0.55
    lats[lats < lat_min - padding_lat] = None
    lats[lats > lat_max + padding_lat] = None
    lons[lons < lon_min - padding_lon] = None
    lons[lons > lon_max + padding_lon] = None

    grid = []
    grid_idx = []

    for lon_idx, lat in enumerate(lats):
        for lat_idx, lon in enumerate(lons):
            if lat == None:
                continue
            if lon == None:
                continue

            grid.append((lon, lat))
            grid_idx.append((lon_idx, lat_idx))

    return grid, grid_idx


def extract_cru_for_location(lon, lat, groupname, hdf5):

    for ds_var in ['tmp', 'vap', 'pet', 'pre']:
        if not CACHE.get(ds_var):
            fill_cache(ds_var)

        lats = numpy.array(CACHE[ds_var]['lats'])
        lons = numpy.array(CACHE[ds_var]['lons'])

        lat_idx = numpy.abs(lats - lat).argmin()
        lon_idx = numpy.abs(lons - lon).argmin()
        ds = CACHE[ds_var]['ds']

        values_at_loc = ds[:, lat_idx, lon_idx]

        save_location(
            lon, lat, ds_var, values_at_loc, hdf5, groupname, old=True)


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


def draw_basemap(nc_ds, dt_time, lons, lats, nc_var):
    """Plot of global temperature on our random day"""
    #
    fig = pyplot.figure()

    # fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
    # for other projections.
    # m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
    m = Basemap(projection='cyl', resolution='c', lon_0=0)

    m.drawcoastlines()
    m.drawmapboundary()
    time_idx = conf['time_idx']
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


def set_dataset(hdf, groupname, data):
    """replace or set data in groupname of hdf file"""
    if groupname in hdf.keys():
        del hdf[groupname]
    return hdf.create_dataset(groupname, data=data)


def save_location(lat, lon, ds_var, values, hdf5, groupname=None, old=False):
    """Store cru data into hdf5 file
    :param lat: lat used by lai
    :param lon: lon used by lai
    :param x:  lon index
    :param y:  lat index
    :param ds: the nc dataset with x year data
    :param ds_var: nc label of dataset
    :param hdf5: the hdf5 file opbject
    :param old oldstyle group name
    :return: Nothing
    """
    if not groupname:
        groupname = conf['groupname']

    cru_groupname = f"{groupname}/{lon}:{lat}/{ds_var}"

    if old:
        cru_groupname = f"{groupname}/{ds_var}"


    nc_matrix = np.array(values)
    )

    h5ds = set_dataset(hdf5, cru_groupname, nc_matrix)

    # store meta.
    h5ds.attrs['cru_loc'] = [lon, lat]
    h5ds.attrs['cru_idx'] = [x, y]
    log.debug(f'Saved CRU {cru_groupname}')


def main():
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, 'a') as data_file:
        for groupname, location in locations.items():
            lon = location['lon']
            lat = location['lat']
            extract_cru_for_location(lon, lat, groupname, data_file)


if __name__ == '__main__':
    main()

