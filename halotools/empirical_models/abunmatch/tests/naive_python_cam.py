""" Naive python implementation of bin-free conditional abundance matching
"""
import numpy as np


def sample2_window_indices(ix1, x_sample1, x_sample2, nwin):
    """ For the point x1 = x_sample1[ix1], determine the indices of
    the window surrounding each point in sample 2 that defines the
    conditional probability distribution for `ynew`.
    """
    nhalfwin = int(nwin/2)
    npts2 = len(x_sample2)

    x1 = x_sample1[ix1]
    iy2 = min(np.searchsorted(x_sample2, x1), npts2-1)

    if iy2 <= nhalfwin:
        init_iy2_low, init_iy2_high = 0, nwin
    elif iy2 >= npts2 - nhalfwin - 1:
        init_iy2_low, init_iy2_high = npts2-nwin, npts2
    else:
        init_iy2_low = iy2 - nhalfwin
        init_iy2_high = init_iy2_low+nwin

    return init_iy2_low, init_iy2_high


def pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin):
    """ Naive algorithm for implementing bin-free conditional abundance matching
    for use in unit-testing.
    """
    result = np.zeros_like(x_sample1)

    n1 = len(x_sample1)

    for i in range(n1):
        low, high = sample2_window_indices(i, x_sample1, x_sample2, nwin)
        sorted_window = np.sort(y_sample2[low:high])
        rank1 = ranks_sample1[i]
        result[i] = sorted_window[rank1]

    return result
