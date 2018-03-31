#!/usr/bin/env python

import numpy as np
import time
import numpy
import random
import logging
import matplotlib
from sympy.utilities.lambdify import MATH

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

matrix_size = 500

from matplotlib import pyplot as plt


def run_map(value_generator, background_data):
    """
    Display the simulation using matplotlib
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    rw = value_generator()
    plt.show(False)
    plt.draw()

    tic = time.time()
    niter = 0

    for x, y, v in rw:
        niter += 1
        background_data[x][y] = v
        # restore background
        if niter % 5 != 0:
            continue

        ax.clear()
        ax.imshow(background_data, cmap='gray')
        #fig.canvas.restore_region(background)
        #plt.imshow(background_data, interpolation='nearest')

        # redraw just the points
        # fill in the axes rectangle
        # fig.canvas.blit(ax.bbox)
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