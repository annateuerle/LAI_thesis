""""
Plot of the all datasets (cru and lai).
Uses the predictive function below and calculates rmse.
"""
from typing import Dict, Any, Union

# import prepare_data
import logging
from load_datasets import load_data
# from load_datasets import calculate_moving_mean

from matplotlib import pyplot
import numpy as np
from functions_pred_lai import prediction_options
from settings import settings
from settings import locations


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def plot(timestamps, datasets, predictors=None, p_label=None):
    TENYEAR = 120

    if not p_label:
        p_label = settings['prediction_option']

    ds_to_plot = {}

    if not predictors:
        predictors = ['lai', 'tmp', 'pre', 'vap', 'pet']

    time_x = timestamps[:120]

    y_lai_values = datasets['lai'][:TENYEAR]

    for ds_var in predictors:
        ds_to_plot[ds_var] = datasets[ds_var][:TENYEAR]

    y_pred_lai = datasets[f'pred_{p_label}'][:TENYEAR]

    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(5, sharex=True, sharey=False)
    pyplot.title(f"LAI for 2001-2010 'pred_{p_label}' Monthly", y=5.08)

    x = time_x
    lai, = ax1.plot(x, y_lai_values, label='lai')
    pred, = ax1.plot(x, y_pred_lai, color='g', label='pred')

    handles = []

    ax2.set_ylabel('C')
    if 'tmp' in predictors:
        y_tmp_values = ds_to_plot['tmp']
        tmp, = ax2.plot(x, y_tmp_values, color='r', label='T')
        handles.append(tmp)
    if 'tmp_gdd' in predictors:
        y_tmp_values = ds_to_plot['tmp_gdd']
        tmp2, = ax2.plot(x, y_tmp_values, color='orange', label='T2')
        handles.append(tmp2)

    ax3.set_ylabel('mm')
    if 'pre' in predictors:
        y_pre_values = ds_to_plot['pre']
        pre, = ax3.plot(x, y_pre_values, color='b', label='P')
        handles.append(pre)

    if 'pre_one' in predictors:
        pre_one = ds_to_plot['pre_one']
        pre2, = ax3.plot(x, pre_one, color='orange', label='T2')
        handles.append(pre2)

    ax4.set_ylabel('hPa')
    if 'vap' in predictors:
        y_vap_values = ds_to_plot['vap']
        vap, = ax4.plot(x, y_vap_values, color='y', label='V')
        handles.append(vap)

    # vap2, = ax4.plot(x[8:], y_vap_avg_values[8:], color='orange', label='T2')
    ax5.set_ylabel('mm')
    if 'pet' in predictors:
        y_pet_values = ds_to_plot['pet']
        pet, = ax5.plot(x, y_pet_values, label='PE')
        handles.append(pet)

    # pet2, = ax5.plot(x[8:], y_pet_avg_values[8:], color='orange', label='T2')

    pyplot.legend(handles=handles, bbox_to_anchor=(1.1, 1.05))

    # units
    pyplot.xlabel('Time (Months)')

    for txt in pyplot.gca().xaxis.get_majorticklabels():
        txt.set_rotation(90)

    pyplot.tight_layout(h_pad=1.0, pad=2.6, w_pad=1.5)

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    pyplot.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    rmse = calc_rmse(datasets[f'pred_{p_label}'][:120], datasets['lai'][:120])

    pyplot.figtext(
        0.83, 0.84, f'rmse {rmse:.4f}', fontsize=10, horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
    )

    pyplot.show()


def make_prediction(datasets):
    label = settings['prediction_option']
    prediction_function = prediction_options[label]
    pred_lai = prediction_function(datasets)
    datasets[f'pred_{label}'] = pred_lai


def calc_rmse(predictions, targets):

    if type(predictions) == list:
        predictions = np.array(predictions)
        targets = np.array(targets)

    differences = predictions - targets                       # the DIFFERENCES.
    differences_squared = differences ** 2                    # the SQUARES of ^
    mean_of_differences_squared = differences_squared.mean()  # the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           # ROOT of ^
    return rmse_val


def main():
    timestamps, datasets = load_data(settings['groupname'])
    # calculate_moving_mean()
    make_prediction(datasets)
    plot(timestamps, datasets)

if __name__ == '__main__':
    main()

