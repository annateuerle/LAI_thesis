from matplotlib.backend_bases import LocationEvent

settings = {
    # squares we extract
    'DELTA': 2,

    # longnitude and latitude of the location.
    'LON': -94.6572,  # X
    'LAT': 16.6515, # Y

    'hdf_dir':'D:/LAI_thesis/Mexico_2001_2010/*.hdf',
    'groupname': "mexico",
    'hdf5storage': 'lai_cru.hdf5',
    'X': None,
    'Y': None,

    'startyear': 2001,
    'endyear': 2010,
    'ncvar': 'vap',
    'time_idx': 42  # some random month counting from the startyear
}

