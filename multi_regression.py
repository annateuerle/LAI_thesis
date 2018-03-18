#Script to predict LAI based on the all cliamtic variables. Fit a line, y = ax + by + cz + dq + constant

from load_datasets import load_data
from load_datasets import calculate_moving_mean
import logging
import numpy
import predictive_models

from functions_pred_lai import prediction_options
from settings import settings

from predictive_models import calc_rmse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def solver_function_multi(datasets):
    """
    Fit a line, y = ax + by + cz + dq + constant, through some noisy data-points

    :param datasets:
    :return:  best symbol values and rmse for prediction function in settings.
    """
    measurements = []

    for ds_key in ['tmp', 'pre', 'vap', 'pet']:
        input_ds = datasets[ds_key]
        input_ar = numpy.array(input_ds)
        measurements.append(input_ar)

    y1 = datasets['lai']
    y = numpy.array(y1)

    measurements.append(numpy.ones(len(y)))

    # We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]].
    # Now use lstsq to solve for p:
    A = numpy.vstack(measurements).T  # [[x  y z q 1]]

    a, b, c, d, k = numpy.linalg.lstsq(A, y, rcond=None)[0]

    log.info(f'a:{a} b:{b} c:{c} d:{d} k:{k}')

    m = measurements

    y_pred = a * m[0] + b * m[1] + c * m[2] + d * m[3] + k

    calc_rmse(y, y_pred)

    datasets[f'pred_all'] = y_pred

    predictive_models.plot(timestamps, datasets)


if __name__ == '__main__':
    # load hdf5 measurement data.
    timestamps, datasets = load_data()
    # calculate_moving_mean()

    #solver_function(datasets)
    solver_function_multi(datasets)