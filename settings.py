from matplotlib.backend_bases import LocationEvent

settings = {
    # squares we extract
    'DELTA': 1,

    # longnitude and latitude of the location.
    'LON': -92.4491,  # X
    'LAT': 33.2985, # Y

    'hdf_dir':'D:/LAI_thesis/Mala_2001_2010/*.hdf',
    #'groupname': "mala",
    #'groupname': "usa",
    #'groupname': "amazon",
    'groupname': "german_forest",
    'hdf5storage': 'lai_cru.hdf5',
    'X': None,
    'Y': None,

    'startyear': 2001,
    'endyear': 2010,
    'ncvar': 'tmp',
    'time_idx': 42,  # some random month counting from the startyear
    # 'prediction_function': lai_pred_tmp,
    # options, 'vap', 'tmp', 'pre', 'pet',
    'prediction_option': 'tmp',
    'normalize': True,
    'moving_average_months': 0,

}

