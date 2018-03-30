#!/usr/bin/env python

import numpy as np
import time
import matplotlib
# matplotlib.use('GTKAgg')

from matplotlib import pyplot as plt


from extract_green import


def run(niter=1000):
    """
    Display the simulation using matplotlib
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.hold(True)
    rw = randomwalk()
    x, y = next(rw)

    plt.show(False)
    plt.draw()

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(x, y, 'o')[0]
    tic = time.time()

    for ii in range(niter):

        # update the xy data
        x, y = next(rw)
        points.set_data(x, y)

        # restore background
        fig.canvas.restore_region(background)

        # redraw just the points
        ax.draw_artist(points)

        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)

    plt.close(fig)

    print("Average FPS: %.2f" % (niter / (time.time() - tic)))

if __name__ == '__main__':
    run()