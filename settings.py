from matplotlib.backend_bases import LocationEvent

settings = {
    # squares we extract
    'DELTA': 2,

    # longnitude and latitude of the location.
    'LON': -66.295,  # X
    'LAT': -5.709000, # Y

    'hdf_dir':'D:/LAI_thesis/Amazon_2001_2010/*.hdf',
    'groupname': "amazon",
    'hdf5storage': 'lai_cru.hdf5',
    'X': None,
    'Y': None,

    'startyear': 2001,
    'endyear': 2010,
    'ncvar': 'pet',
    'time_idx': 42  # some random month counting from the startyear
}

