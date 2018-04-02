#!/usr/bin/env python

import numpy as np
import time
import numpy
import random
import logging
import matplotlib as mpl
from sympy.utilities.lambdify import MATH
mpl.use('GTkAgg')


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

matrix_size = 500

from matplotlib import pyplot as plt

CMAP = mpl.colors.ListedColormap([
    'gray',  # water.
    'green', # todo
    'red',   # loaded
    'darkgray', # not interesting.
])



def run_map(value_generator, background_data, green, modulo=5):
    """
    Display the simulation using matplotlib
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    rw = value_generator()
    plt.show(False)
    plt.draw()

    background_data[green.mask] = 1
    background_data[background_data > 5] = 3
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)

    tic = time.time()
    niter = 0

    for x, y, v in rw:
        niter += 1
        background_data[x][y] = v

        if niter % modulo != 0:
            continue

        ax.clear()
        ax.imshow(background_data, norm=norm, cmap=CMAP)
        fig.canvas.draw()
        log.info(niter)

    plt.close(fig)

    print("Average FPS: %.2f" % (niter / (time.time() - tic)))


def generate_random_dots():
    for x in range(500):
        x = random.randrange(MATRIX_SIZE)
        y = random.randrange(MATRIX_SIZE)
        yield x, y, 10


if __name__ == '__main__':
    MATRIX_SIZE = 500
    background = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE))
    run_map(generate_random_dots, background)