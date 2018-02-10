import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def lai_pred_tmp(datasets, symbols=[0.1, 0.5]):
    """
    make a prediction of lai with temperature
    :param datasets:
    :return: list with lai predictions
    """
    ds_tmp = datasets['tmp']
    predictions = []
    for tmp in ds_tmp:
        prediction = symbols[0] * tmp + symbols[1]
        predictions.append(prediction)

    return predictions

def lai_pred_pre(datasets, symbols=[0.1, 0.5]):
    """
    make a prediction of lai with vapour pressure
    :param datasets:
    :return: list with lai predictions
    """
    ds_pre = datasets['pre']
    predictions = []
    for pre in ds_pre:
        prediction = symbols[0] * pre + symbols[1]
        predictions.append(prediction)

    return predictions

def lai_pred_vap(datasets, symbols=[0.1, 0.5]):
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

def lai_pred_pet(datasets, symbols=[0.1, 0.5]):
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
