
from load_datasets import load_data
import logging

from functions_pred_lai import prediction_options

from settings import settings

from predictive_models import calc_rmse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def calculate_rmse_symbols(symbols, datasets) -> float:
    """
    :param symbols: x, y , z..
    :param datasets:
    :return: rmse of prediction function.
    """
    pred_to_solve = settings['prediction_option']
    pred_function = prediction_options[pred_to_solve]
    measured_lai = datasets['lai']

    test_predicted = pred_function(datasets, symbols=symbols)
    rmse = calc_rmse(test_predicted, measured_lai)
    return rmse


def try_symbols(test_symbols, low_rmse, symbols, datasets) -> (float, list):
    """
    return lowest rmse and symbols of given arguments.

    :param test_symbols: x,y z we are testing
    :param low_rmse: lowes rmse till now
    :param symbols: lowest symbols x, y , z till now
    :param datasets: all hdf5 data
    :return: lowest_rmse, lowest_symbols
    """

    rmse = calculate_rmse_symbols(test_symbols, datasets)


    if rmse < low_rmse:
        log.debug('Found new low RMSE %s', test_symbols)
        return rmse, test_symbols

    return low_rmse, symbols


def solve_big_area(datasets, constraints=[(0,10), (0,10)]) -> list:
    """
    Scan big part of solution area and return values
    'rmse' values for symbol options.

    for now only checks symbol values between 0-10.

    :param datasets: all hdf data
    :param constraints: range of symbol values
    :return: list of tuples with [(rmse, symbols)..]
    """
    options = []
    x1, x2 = constraints[0]
    y1, y2 = constraints[1]
    for tx1 in range(x1, x2):
        for tx2 in range(y1, y2):
            test_symbols = [tx1, tx2]
            rmse = calculate_rmse_symbols(test_symbols, datasets)
            options.append((rmse, test_symbols))

    return options

def solve_options(datasets, options):
    """
    Take the best x options and search within
    one option in a 100x100 resolution.

    so for option (3,1) we search all values from
    [3, 1] to [4, 2] in steps of 0.01.

    :param datasets:
    :param options: list with scores and options.
    :return: (lowest rmse , lowest_symbols)
    """

    options.sort()
    best_options = options[:5]

    symbols = [0, 0]
    rmse = 10000

    for score, o_symbols in best_options:
        log.debug(f'testing option: {o_symbols} best: {rmse}')
        p1, p2 = o_symbols

        for i in range(101):
            p1f = p1 + i * 0.01
            for j in range(101):
                p2f = p2 + j * 0.01
                test_sym = [p1f, p2f]
                rmse, symbols = try_symbols(test_sym, rmse, symbols, datasets)

    return rmse, symbols


def solver_function(datasets):
    """
    :param datasets:
    :return:  best symbol values and rmse for prediction function in settings.
    """

    # find options in symbols spcae with rmse score.
    options = solve_big_area(datasets, constraints=[(0, 10), (0, 10)])
    # deep seearch in the best options for optimal symbol values.
    rmse, symbols = solve_options(datasets, options)

    pred_to_solve = settings['prediction_option']
    log.debug(f'RMSE lowest {rmse} for pred {pred_to_solve} symbols {symbols}')


if __name__ == '__main__':
    # load hdf5 measurement data.
    timestamps, datasets = load_data()
    solver_function(datasets)