from settings import settings
import h5py
from datetime import date

timsetamps = None

datasets = {
    'lai': None,
    'tmp': None,
    'pre': None,
    'vap': None,
    'pet': None,
}


def load_data():
    storage_name = settings['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:

        x_time = data_file['timestamps']

        time_x = []
        for t in x_time:
            dt = date.fromtimestamp(t)
            time_x.append(dt)

        global timestamps
        timestamps = time_x[:120]
        groupname = settings['groupname']
        for var in datasets.keys():
            datasets[var] = list(data_file[f'{groupname}-{var}'])[:120]


    return timestamps, datasets