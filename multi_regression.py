#Script to predict LAI based on the all cliamtic variables. Fit a line, y = ax + by + cz + dq + constant

from load_datasets import load_data
from load_datasets import normalized_dataset
import logging
import numpy
import predictive_models
import math


from predictive_models import calc_rmse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def solver_function_multi(datasets, timestamps, predictors=('tmp', 'pre', 'vap', 'pet'), label='all'):
    """
    Fit a line, y = ax + by + cz + dq + constant, through some noisy data-points

    :param datasets:  the source data
    :param timestamps datetimes of time period. used to make graph.
    :param predictors  the dataset names we use to predict.
    :param label  we store this predicted lai under pred_{label}
    :return:  best symbol values and rmse for prediction function in settings.
    """
    measurements = []
    plot_predictors = []

    for ds_key in predictors:
        if type(ds_key) is str:
            input_ds = datasets[ds_key]
            input_ar = numpy.array(input_ds)
            plot_predictors.append(ds_key)
        else:
            input_ar = ds_key(datasets)
            plot_predictors.append(ds_key.__name__)

        measurements.append(input_ar)

    y1 = datasets['lai']
    y = numpy.array(y1)

    measurements.append(numpy.ones(len(y)))

    # We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]].
    # Now use lstsq to solve for p:
    A = numpy.vstack(measurements).T  # [[x  y z q 1]]

    parameters = numpy.linalg.lstsq(A, y, rcond=None)[0]

    log.info(f'parameters: %s', list(zip(predictors, parameters)))

    m = measurements

    y_pred = numpy.zeros(120)

    for i, p in enumerate(parameters[:-1]):   # we skip K.
        y_pred += p * m[i]

    calc_rmse(y, y_pred)

    datasets[f'pred_{label}'] = y_pred

    predictive_models.plot(timestamps, datasets, plot_predictors, p_label=label)


def make_models(models_to_make, datasets, timestamps):
    for label, p_labels in models_to_make.items():
        solver_function_multi(datasets, timestamps, p_labels, label=label)


def aic_criterion(models_to_make, datasets):
    # load hdf5 measurement data.
    lai = datasets['lai']
    for p, ds_label in models_to_make.items():
        p_label = f'pred_{p}'
        predicted_lai = datasets[p_label]
        R = numpy.square(lai - predicted_lai).sum()
        # print(R)
        m = len(ds_label) # len variables
        n = len(lai)  # measurements
        A = n * math.log((2*math.pi)/n) + n + 2 + n * math.log(R) + 2 * m
        print('%s %.4f' % (p, A))


def tmp_gdd(datasets):
    tmp = datasets['tmp_original']
    tmp[tmp < 5] = 0
    n_5_tmp = normalized_dataset(tmp)
    datasets['tmp_gdd'] = n_5_tmp
    return n_5_tmp


def pre_one(datasets):
    preone = numpy.roll(datasets['pre'], 1)
    datasets['pre_one'] = preone
    return preone


def main():
    # load hdf5 measurement data.
    timestamps, datasets = load_data()
    models_options = {
        # 'p4': ['tmp', 'pre', 'pet', 'vap'],
        # 'p3': ['tmp', 'pre', 'pet'],
        # 'p2': ['tmp', 'pre',],
        'gdd': [tmp_gdd, pre_one, 'pre', 'pet'],
        'gdd2': [tmp_gdd, pre_one],
        'gdd3': [tmp_gdd],
        'gdd4': [tmp_gdd, 'tmp'],
        'gdd5': [tmp_gdd, 'tmp', 'vap'],
        'p1-t': ['tmp'],
        # 'p1-v': ['vap'],
        'p2-tv': ['tmp', 'vap'],
    }
    make_models(models_options, datasets, timestamps)
    aic_criterion(models_options, datasets)


if __name__ == '__main__':
    main()

