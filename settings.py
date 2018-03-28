""""
The file to describe all the settings used in the files.
"""
from matplotlib.backend_bases import LocationEvent

settings = {
    #hdf file
    'hdf_dir': 'D:/LAI_thesis/Mala_2001_2010/*.hdf',
    #'hdf_dir': 'D:/LAI_thesis/Amazon_2001_2010/*.hdf',
    #'hdf_dir': 'D:/LAI_thesis/USA_2001_2010/*.hdf',

    #groupname of five locations
     #'groupname': "mala",
    #'groupname': "mexico",
    #'groupname': "usa",
    #'groupname': "amazon",
    'groupname': "german_forest",

    'hdf5storage': 'lai_cru.hdf5',
    'X': None,
    'Y': None,

    #time settings
    'startyear': 2001,
    'endyear': 2010,
    'ncvar': 'tmp',
    'time_idx': 42,  # some random month counting from the startyear (for the CRU map)

    # 'prediction_function': lai_pred_tmp,

    #prediction options, 'vap', 'tmp', 'pre', 'pet', 'all',
    'prediction_option': 'all',
    'normalize': True,  # avg 0.
    'moving_average_months': 0,

}

locations = {
    # location: lat lon
    'mala': {'lat': 4.819, 'lon': 102.500},
    'german_forest': {'lat': 51.1156, 'lon': 7.5046},
    'amazon': {'lat': -5.709000, 'lon': -66.295},
    'mexico': {'lat': 16.6515, 'lon': -94.6572},
    'usa': {'lat': 33.2985, 'lon': -92.4491},
}

"""
1. Kierspe, Germany
MCD12Q1 CLASS/Band: 4 (Deciduous Broadleaf Trees)
Lat Lon (51.1156, 7.5046)
MODIS_NL_2001_2010
groupname german_forest

2. Tapaua, State of Amazonas, Brazil
MCD12Q1 CLASS/Band: 2 (Evergreen Broadleaf Trees)
Lat Lon (-5.709000, -66.295)
Amazon_2001_2010
groupname amazon

3. San Miguel Chimalapa, Oax., Mexico
MCD12Q1 CLASS/Band: 4
Lat Lon (16.6515, -94.6572)
Mexico_2001_2010
groupname mexico

4. Township, AR, USA
MCD12Q1 CLASS/Band: 4
Lat Lon (33.2985, -92.4491)
USA_2001_2010
groupname usa

5. Chiku, Kelantan, Malaysia
MCD12Q1 CLASS/Band: 2
Lat Lon (4.819, 102.500)
Mala_2001_2010
groupname mala

"""