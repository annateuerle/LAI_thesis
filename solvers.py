
from load_datasets import load_data
import logging

from functions_pred_lai import prediction_options

from settings import settings

from predictive_models import calc_rmse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def try_symbols(test_symbols, low_rmse, symbols):

    pred_to_solve = settings['prediction_option']
    pred_function = prediction_options[pred_to_solve]
    measured_lai = datasets['lai']

    log.debug(f'Trying {test_symbols}')

    test_predicted = pred_function(datasets, test_symbols)
    rmse = calc_rmse(test_predicted, measured_lai)

    if rmse < low_rmse:
        log.debug('Found new low RMSE %s', symbols)
        return rmse, test_symbols

    return low_rmse, symbols


def solver_function(datasets):

    symbols = [0, 0]
    rmse = 10000

    for p1 in range(0, 11):
        for p2 in range(0, 11):
            test_sym = [p1, p2]
            rmse, symbols = try_symbols(test_sym, rmse ,symbols)

    p1, p2 = symbols

    for i in range(101):
        p1f = p1 + i * 0.01
        for j in range(101):
            p2f = p2 + j * 0.01
            test_sym = [p1f, p2f]
            rmse, symbols = try_symbols(test_sym, rmse, symbols)

    pred_to_solve = settings['prediction_option']
    log.debug(f'RMSE lowest {rmse} for pred {pred_to_solve} symbols {symbols}')


if __name__ == '__main__':
    # load hdf5 measurement data.
    timestamps, datasets = load_data()
    solver_function(datasets)