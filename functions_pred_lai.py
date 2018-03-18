#Predicts the value for m and c in the shape y=mx+c where y is LAI dataset and x is cliamtic variable.
#If in the settings is defined moving average, then it takes moving average lai.

import logging
from settings import settings

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def get_dataset(datasets, ds_var):
    """
    If there is a moving average month defined in the settings.
    use that one.

    :param datasets:
    :param ds_var:
    :return: dataset to make prediction with.
    """

    ds_tmp = datasets['tmp']

    if settings.get('moving_average_months'):
        months = settings['moving_average_months']
        ds_tmp = datasets[f'{ds_var}_moving_avg_{months}']

    return ds_tmp


def lai_pred_tmp(datasets, symbols=[0.09, 0.369]):
    """
    make a prediction of lai with temperature
    :param datasets:
    :return: list with lai predictions
    """
    # m -0.4867620174362727, c: 17.4597085339703

    ds_tmp = get_dataset(datasets, 'tmp')

    predictions = []
    for tmp in ds_tmp:
        prediction = symbols[0] * tmp + symbols[1]
        predictions.append(prediction)

    return predictions

def lai_pred_pre(datasets, symbols=[0.00, 4.46]):
    """
    make a prediction of lai with vapour pressure
    :param datasets:
    :return: list with lai predictions
    """
    ds_pre = get_dataset(datasets, 'pre')

    predictions = []
    for pre in ds_pre:
        prediction = symbols[0] * pre + symbols[1]
        predictions.append(prediction)

    return predictions

def lai_pred_vap(datasets, symbols=[0.09, 2.0]):
    """
    make a prediction of lai with vapour pressure
    :param datasets:
    :return: list with lai predictions
    """
    ds_vap = datasets['vap']
    predictions = []
    for vap in ds_vap:
        prediction = symbols[0] * vap + symbols[1]
        predictions.append(prediction)

    return predictions

def lai_pred_pet(datasets, symbols=[1.2, 1.0]):
    """
    make a prediction of lai with vapour pressure
    :param datasets:
    :return: list with lai predictions
    """
    ds_pet = datasets['pet']
    predictions = []
    for pet in ds_pet:
        prediction = symbols[0] * pet + symbols[1]
        predictions.append(prediction)

    return predictions


prediction_options = {
    'tmp': lai_pred_tmp,
    'pre': lai_pred_pre,
    'vap': lai_pred_vap,
    'pet': lai_pred_pet,
}
