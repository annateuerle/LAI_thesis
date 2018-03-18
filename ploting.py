"""
Plot to compare Svitzky-Golay filter with the original datasets.
"""

import load_datasets
import numpy as np
from matplotlib import pyplot
from predictive_models import calc_rmse


def plot_filter(timestamps, datasets):

    time_x = timestamps[:120]

    y_lai_values = datasets['lai'][:120]
    #width and the polynomial in the filter
    f_y_lai_values1 = load_datasets.savitzky_golay(y_lai_values, 9, 4)
    #f_y_lai_values2 = load_datasets.savitzky_golay(y_lai_values, 7, 2)

    rmse = calc_rmse(f_y_lai_values1[:120], datasets['lai'][:120])

    pyplot.figtext(
        0.83, 0.84, f'rmse {rmse:.4f}', fontsize=10, horizontalalignment='center',
        verticalalignment='center', bbox=dict(facecolor='white', alpha=1),
    )
    pyplot.plot(time_x, y_lai_values, label='Original LAI dataset')
    pyplot.plot(time_x, f_y_lai_values1, label='Filtered LAI dataset')
    #pyplot.plot(time_x, f_y_lai_values2, label=2)

    pyplot.title('Application of Savitzky-Goley filter')
    pyplot.ylabel('LAI')
    pyplot.xlabel('Time (Months)')
    pyplot.legend()
    pyplot.show()



if __name__ == '__main__':
    timestamps, datasets = load_datasets.load_data()
    plot_filter(timestamps, datasets)
