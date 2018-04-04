#!/usr/bin/env python

import numpy as np
import time
import numpy
import numpy.ma as ma
import random
import logging
import matplotlib as mpl
from sympy.utilities.lambdify import MATH

# mpl.use('GTk3Agg')
# mpl.use('Qt5Agg')
mpl.rc('figure', figsize=(8, 8))


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

matrix_size = 500

from matplotlib import pyplot as plt

CMAP = mpl.colors.ListedColormap([
    'gray',  # water.
    'green', # to calculate with
    'red',   # loaded
    'darkgray', # not interesting.
])



def run_map(value_generator, background_data, green, grid, modulo=5):
    """
    Display the simulation using matplotlib
    """

    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    ax.set_aspect('equal')
    rw = value_generator()
    plt.show(False)
    # plt.draw()
    # plt.figure()

    background_data[green] = 1
    background_data[background_data > 5] = 3
    norm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)

    tic = time.time()
    niter = 0

    def map_update():

        ax.clear()
        ax.imshow(background_data, norm=norm, cmap=CMAP)

        for x, y in grid:
            plt.plot(x, y, '+b')

        fig.canvas.draw()
        log.info(niter)
        plt.pause(10.001)  # I ain't needed!!!

    for x, y, v in rw:
        niter += 1
        background_data[x][y] = v

        if niter % modulo != 0:
            continue

        map_update()

    map_update()

    plt.show()
    print("Average FPS: %.2f" % (niter / (time.time() - tic)))


def green_simulator(mask):
    def generate_dots():
        xarr, yarr = numpy.where(mask)
        for x, y in zip(xarr, yarr):
            yield x, y, 2

    return generate_dots


# test this plotting code.
if __name__ == '__main__':
    MATRIX_SIZE = 100
    # random map 11 ground types
    # background = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE))
    background = numpy.random.randint(0, 4, size=(MATRIX_SIZE, MATRIX_SIZE))
    # make 1's background data
    gray = ma.masked_inside(background, 1, 1)
    background[gray.mask] = 4
    # make 3's also 0 for water.
    water = ma.masked_inside(background, 3, 3)
    background[water.mask] = 0
    green = ma.masked_inside(background, 2, 2)
    generator = green_simulator(green)
    # Test the progress plotting of a map
    run_map(generator, background, green.mask, [], modulo=50)
