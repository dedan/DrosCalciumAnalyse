
import itertools as it
import numpy as np

def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect*0.3)


def select_n_channels(data, n):
    """ select n rows which have the most possible odors (columns) in common

        Not for all animals the same odors are available because sometimes they
        are not consistend over stimulus repetition and are therefore sorted out.
        This function helps to find the subset of odors which are available for
        n out of size(data, 0) animals.
    """
    best = -1
    for comb in it.combinations(range(np.size(data, 0)), n):
        s = (np.sum(data[comb, :], 0) == n).astype('int')
        if np.sum(s) > best:
            best = np.sum(s)
            best_comb = comb
    return best_comb
