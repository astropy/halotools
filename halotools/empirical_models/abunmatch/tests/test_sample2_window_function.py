""" Module testing the sample2_window_indices function that returns the
relevant CAM window to the naive python implementation.
"""
import numpy as np
from .naive_python_cam import sample2_window_indices


def test_left_edge_window():
    """ Setup: x1 == x2. Enforce proper behavior at the leftmost edge.
    """
    n1, n2 = 20, 20
    x_sample1 = np.arange(n1)
    x_sample2 = np.arange(n2)

    nwin = 5

    ix1 = 0
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (0, nwin)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 1
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (0, nwin)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 2
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (0, nwin)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 3
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (1, nwin+1)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 4
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (2, nwin+2)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin


def test_right_edge_window():
    """ Setup: x1 == x2. Enforce proper behavior at the rightmost edge.
    """
    n1, n2 = 20, 20
    x_sample1 = np.arange(n1)
    x_sample2 = np.arange(n2)

    nwin = 5

    ix1 = 19
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (n2-nwin, n2)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 18
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (n2-nwin, n2)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 17
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (n2-nwin, n2)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 16
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (n2-nwin-1, n2-1)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin

    ix1 = 15
    init_iy2_low, init_iy2_high = sample2_window_indices(
        ix1, x_sample1, x_sample2, nwin)
    assert (init_iy2_low, init_iy2_high) == (n2-nwin-2, n2-2)
    assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin


def test_all_x1_less_than_x2():
    """ Setup: np.all(x1 < x2.min()).

    Enforce proper behavior at the leftmost edge.
    """
    n1, n2 = 20, 20
    x_sample1 = np.arange(n1)
    x_sample2 = np.arange(100, 100+n2)

    nwin = 5

    for ix1 in range(n1):
        init_iy2_low, init_iy2_high = sample2_window_indices(
            ix1, x_sample1, x_sample2, nwin)
        assert (init_iy2_low, init_iy2_high) == (0, nwin), "ix1 = {0}".format(ix1)
        assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin


def test_all_x1_greater_than_x2():
    """ Setup: np.all(x1 < x2.min()).

    Enforce proper behavior at the leftmost edge.
    """
    n1, n2 = 20, 20
    x_sample1 = np.arange(n1)
    x_sample2 = np.arange(-100, -100+n2)

    nwin = 5

    for ix1 in range(n1):
        init_iy2_low, init_iy2_high = sample2_window_indices(
            ix1, x_sample1, x_sample2, nwin)
        assert (init_iy2_low, init_iy2_high) == (n2-nwin, n2), "ix1 = {0}".format(ix1)
        assert len(x_sample2[init_iy2_low:init_iy2_high]) == nwin
