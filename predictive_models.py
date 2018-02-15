import time_series
import logging
from load_datasets import load_data

from matplotlib import pyplot
import numpy as np
from functions_pred_lai import prediction_options
from settings import settings
import matplotlib.ticker as mticker


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def plot(timestamps, datasets):
    p_label = settings['prediction_option']

    time_x = timestamps[:120]
    y_lai_values = datasets['lai'][:120]
    y_tmp_values = datasets['tmp'][:120]
    y_pre_values = datasets['pre'][:120]
    y_vap_values = datasets['vap'][:120]
    y_pet_values = datasets['pet'][:120]
    y_pred_lai = datasets[f'pred_{p_label}'][:120]


    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(5, sharex=True, sharey=False)
    pyplot.title(f"LAI for 2001-2010 'pred_{p_label}' Monthly",y=5.08 )
    x = time_x
    lai, = ax1.plot(x, y_lai_values, label='lai')
    pred, = ax1.plot(x, y_pred_lai, color='g', label='pred')

    ax2.set_ylabel('C')
    tmp, = ax2.plot(x, y_tmp_values, color='r', label='T')
    ax3.set_ylabel('mm')
    pre, = ax3.plot(x, y_pre_values, color='b', label='P')
    ax4.set_ylabel('hPa')
    vap, = ax4.plot(x, y_vap_values, color='y', label='V')
    ax5.set_ylabel('mm')
    pet, = ax5.plot(x, y_pet_values, label='PE')

    pyplot.legend(handles=[pred, lai, tmp, pre, vap, pet], bbox_to_anchor=(1.1, 1.05))

    #units
    pyplot.xlabel('Time (Months)')


    for txt in pyplot.gca().xaxis.get_majorticklabels():
        txt.set_rotation(90)

    pyplot.tight_layout(h_pad=1.0,pad=2.6, w_pad=1.5 )

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    pyplot.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    rmse = calc_rmse(datasets[f'pred_{p_label}'][:120], datasets['lai'][:120])

    pyplot.figtext(
        0.83, 0.84, f'rmse {rmse:.4f}', fontsize=10, horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
    )

    #fig = pyplot.figure()
    #textplot = fig.add_subplot(111)
    pyplot.show()

def make_prediction(datasets):
    label = settings['prediction_option']
    prediction_function = prediction_options[label]
    pred_lai = prediction_function(datasets)
    datasets[f'pred_{label}'] = pred_lai

#def lai_pred_tmp(tmp):

    #return 0.1 * tmp + 0.5

def calc_rmse(predictions, targets):

    if type(predictions) == list:
        predictions = np.array(predictions)
        targets = np.array(targets)

    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val  #get the ^

if __name__ == '__main__':
    global timestamps
    global datasets
    timestamps, datasets = load_data()
    make_prediction(datasets)
    plot(timestamps, datasets)
