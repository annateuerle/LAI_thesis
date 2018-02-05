from matplotlib.backend_bases import LocationEvent

settings = {
    # squares we extract
    'DELTA': 2,

    # longnitude and latitude of the location.
    'LON': 7.504,  # X
    'LAT': 51.115, # Y

    'hdf_dir':'D:/LAI_thesis/MODIS_NL_2001_2010/*.hdf',
    'groupname': "german_forest",
    'hdf5storage': 'lai_cru.hdf5',
    'X': None,
    'Y': None,

    'startyear': 2001,
    'endyear': 2010,
    'ncvar': 'vap',
    'time_idx': 42  # some random month counting from the startyear
}

