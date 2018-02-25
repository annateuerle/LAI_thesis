from settings import settings
import h5py
import logging
from datetime import date

timsetamps = None

datasets = {
    'lai': None,
    'tmp': None,
    'pre': None,
    'vap': None,
    'pet': None,
}

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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


def calculate_moving_mean():
    """

    :return: a plot.
    """
    moving_avg = []
    ds_var = settings['prediction_option']
    moving_average_result = [0]
    x_months = settings.get('moving_average_months', 0)
    if not x_months:
        log.debug('No moving average defined')
        return

    moving_average_result = (x_months - 1) * [0]

    for value in datasets[ds_var]:
        moving_avg.append(value)

        if len(moving_avg) > x_months:
            # remove oldest value
            moving_avg.pop(0)

        if len(moving_avg) == x_months:
            m_avg = sum(moving_avg) / x_months
            moving_average_result.append(m_avg)

    dataset_label = f'{ds_var}_moving_avg_{x_months}'
    assert len(moving_average_result) == len(datasets[ds_var])
    datasets[dataset_label] = moving_average_result

    #from matplotlib import pyplot

    #pyplot.plot(timestamps[8:], moving_average_result[8:], 'b', timestamps, datasets[ds_var], 'g')
    #pyplot.show()


if __name__ == '__main__':
    timestamps, datasets = load_data()
    calculate_moving_mean()