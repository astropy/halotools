"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest
from ..bin_free_cam import conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .naive_python_cam import pure_python_rank_matching
from ....utils import unsorting_indices


fixed_seed = 43
fixed_seed2 = 44


def test1():
    """ Test case where x and x2 are sorted, y and y2 are sorted,
    and the nearest x2 value is lined up with x
    """
    nwin = 3

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x2 = x

    y = np.arange(1, len(x)+1)
    y2 = y*10.

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))

    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)
    print("ynew  = {0}".format(result.astype('i4')))

    assert np.all(result == y2)


def test2():
    """ Test case where x and x2 are sorted, y and y2 are not sorted,
    and the nearest x2 value is lined up with x
    """
    nwin = 3
    nhalfwin = int(nwin/2)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x2 = x+0.01

    with NumpyRNGContext(fixed_seed):
        y = np.round(np.random.rand(len(x)), 2)
        y2 = np.round(np.random.rand(len(x2)), 2)

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))

    print("ranks1  = {0}".format(cython_sliding_rank(x, y, nwin)))
    print("ranks2  = {0}".format(cython_sliding_rank(x2, y2, nwin)))

    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    print("\n\nynew  = {0}".format(np.abs(result)))
    print("y2    = {0}".format(y2))
    print("y     = {0}\n".format(y))

    #  Test all points except edges
    for itest in range(nhalfwin, len(x)-nhalfwin):
        low = itest-nhalfwin
        high = itest+nhalfwin+1
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[nhalfwin]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        assert itest_result == itest_correct_result

    #  Test left edge
    for itest in range(nhalfwin):
        low, high = 0, nwin
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[itest]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        msg = "itest_result = {0}, correct result = {1}"
        assert itest_result == itest_correct_result, msg.format(
            itest_result, itest_correct_result)

    #  Test right edge
    for iwin in range(nhalfwin+1, nwin):
        itest = iwin + len(x) - nwin
        low, high = len(x)-nwin, len(x)
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[iwin]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        msg = "itest_result = {0}, correct result = {1}"
        assert itest_result == itest_correct_result, msg.format(
            itest_result, itest_correct_result)


def test3():
    """ Test hard-coded case where x and x2 are sorted, y and y2 are sorted,
    but the nearest x--x2 correspondence is no longer simple
    """
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)
    i2_matched = np.array([0, 0, 0, 4, 4])

    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)
    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]

    assert np.allclose(result, correct_result)


def test4():
    """ Regression test for buggy treatment of rightmost endpoint behavior
    """

    n1, n2, nwin = 8, 8, 3
    x = np.round(np.linspace(0.15, 1.3, n1), 2)
    with NumpyRNGContext(fixed_seed):
        y = np.round(np.random.uniform(0, 1, n1), 2)
    ranks_sample1 = cython_sliding_rank(x, y, nwin)

    x2 = np.round(np.linspace(0.15, 1.3, n2), 2)
    with NumpyRNGContext(fixed_seed):
        y2 = np.round(np.random.uniform(-4, -3, n2), 2)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, pure_python_result)


def test_brute_force_interior_points():
    """
    """
    num_tests = 50

    nwin = 11
    nhalfwin = int(nwin/2)

    for i in range(num_tests):
        seed = fixed_seed + i
        with NumpyRNGContext(seed):
            x1_low, x2_low = np.random.uniform(-10, 10, 2)
            x1_high, x2_high = np.random.uniform(100, 200, 2)
            n1, n2 = np.random.randint(30, 100, 2)
            x = np.sort(np.random.uniform(x1_low, x1_high, n1))
            x2 = np.sort(np.random.uniform(x2_low, x2_high, n2))

            y1_low, y2_low = np.random.uniform(-10, 10, 2)
            y1_high, y2_high = np.random.uniform(100, 200, 2)
            y = np.random.uniform(y1_low, y1_high, n1)
            y2 = np.random.uniform(y2_low, y2_high, n2)

        ranks_sample1 = cython_sliding_rank(x, y, nwin)
        ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

        pure_python_result = pure_python_rank_matching(x, ranks_sample1,
                x2, ranks_sample2, y2, nwin)

        cython_result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

        assert np.allclose(pure_python_result[nhalfwin:-nhalfwin],
            cython_result[nhalfwin:-nhalfwin])


def test_brute_force_left_endpoints():
    """
    """

    num_tests = 50

    nwin = 11
    nhalfwin = int(nwin/2)

    for i in range(num_tests):
        seed = fixed_seed + i
        with NumpyRNGContext(seed):
            x1_low, x2_low = np.random.uniform(-10, 10, 2)
            x1_high, x2_high = np.random.uniform(100, 200, 2)
            n1, n2 = np.random.randint(30, 100, 2)
            x = np.sort(np.random.uniform(x1_low, x1_high, n1))
            x2 = np.sort(np.random.uniform(x2_low, x2_high, n2))

            y1_low, y2_low = np.random.uniform(-10, 10, 2)
            y1_high, y2_high = np.random.uniform(100, 200, 2)
            y = np.random.uniform(y1_low, y1_high, n1)
            y2 = np.random.uniform(y2_low, y2_high, n2)

        ranks_sample1 = cython_sliding_rank(x, y, nwin)
        ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

        pure_python_result = pure_python_rank_matching(x, ranks_sample1,
                x2, ranks_sample2, y2, nwin)

        cython_result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

        #  Test left edge
        assert np.allclose(pure_python_result[:nhalfwin], cython_result[:nhalfwin])


def test_brute_force_right_points():
    """
    """

    num_tests = 50

    nwin = 11
    nhalfwin = int(nwin/2)

    for i in range(num_tests):
        seed = fixed_seed + i
        with NumpyRNGContext(seed):
            x1_low, x2_low = np.random.uniform(-10, 10, 2)
            x1_high, x2_high = np.random.uniform(100, 200, 2)
            n1, n2 = np.random.randint(30, 100, 2)
            x = np.sort(np.random.uniform(x1_low, x1_high, n1))
            x2 = np.sort(np.random.uniform(x2_low, x2_high, n2))

            y1_low, y2_low = np.random.uniform(-10, 10, 2)
            y1_high, y2_high = np.random.uniform(100, 200, 2)
            y = np.random.uniform(y1_low, y1_high, n1)
            y2 = np.random.uniform(y2_low, y2_high, n2)

        ranks_sample1 = cython_sliding_rank(x, y, nwin)
        ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

        pure_python_result = pure_python_rank_matching(x, ranks_sample1,
                x2, ranks_sample2, y2, nwin)

        cython_result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

        #  Test right edge
        assert np.allclose(pure_python_result[-nhalfwin:], cython_result[-nhalfwin:])


def test_hard_coded_case1():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    correct_result = [0.03, 0.54, 0.67, 0.73, 0.86]
    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, correct_result)

def test_hard_coded_case2():
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]
    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, correct_result)


def test_hard_coded_case3():
    """ x==x2.

    So the CAM windows are always the same.
    So the first two windows are the leftmost edge,
    the middle entry uses the middle window,
    and the last two entries use the rightmost edge window.
    """
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.04, 0.3, 0.6, 5., 10.]
    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, correct_result)


def test_hard_coded_case5():
    nwin = 3

    x = np.array((1., 1., 1, 1, 1))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.6, 5., 5., 5., 10.]
    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    print("\n\ncorrect result = {0}".format(correct_result))
    print("cython result  = {0}\n".format(result))

    msg = "Cython implementation incorrectly ignores searchsorted result for edges"
    assert np.allclose(result, correct_result), msg


def test_hard_coded_case4():
    """ Every x2 is larger than the largest x.

    So the only CAM window ever used is the first 3 elements of y2.
    """
    nwin = 3

    x = np.array((0., 0., 0., 0., 0.))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.04, 0.3, 0.3, 0.3, 0.6]
    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, correct_result)


def test_hard_coded_case6():
    """
    """

    x = [0.15, 0.31, 0.48, 0.64, 0.81, 0.97, 1.14, 1.3]
    x2 = [0.15, 0.38, 0.61, 0.84, 1.07, 1.3]

    y = [0.22, 0.87, 0.21, 0.92, 0.49, 0.61, 0.77, 0.52]
    y2 = [-3.78, -3.13, -3.79, -3.08, -3.51, -3.39]

    nwin = 5

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    result = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(result, pure_python_result)


def test_subgrid_noise1():
    n1, n2 = int(5e4), int(5e3)

    with NumpyRNGContext(fixed_seed):
        x = np.sort(np.random.uniform(0, 10, n1))
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.sort(np.random.uniform(0, 10, n2))
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 201
    result = conditional_abunmatch(x, y, x2, y2, nwin1, add_subgrid_noise=False)
    result2 = conditional_abunmatch(x, y, x2, y2, nwin1, add_subgrid_noise=True)
    assert np.allclose(result, result2, atol=0.1)
    assert not np.allclose(result, result2, atol=0.02)
    assert np.all(result - result2 != 0)

    nwin2 = 1001
    result = conditional_abunmatch(x, y, x2, y2, nwin2, add_subgrid_noise=False)
    result2 = conditional_abunmatch(x, y, x2, y2, nwin2, add_subgrid_noise=True)
    assert np.allclose(result, result2, atol=0.02)
    assert np.all(result - result2 != 0)


def test_initial_sorting1():
    """
    """
    n1, n2 = int(2e3), int(1e3)

    with NumpyRNGContext(fixed_seed):
        x = np.sort(np.random.uniform(0, 10, n1))
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.sort(np.random.uniform(0, 10, n2))
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 101
    result = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=False, assume_x2_is_sorted=False,
            add_subgrid_noise=False)
    result2 = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=True, assume_x2_is_sorted=True,
            add_subgrid_noise=False)
    assert np.allclose(result, result2)



def test_initial_sorting2():
    """
    """
    n1, n2 = int(2e3), int(1e3)

    with NumpyRNGContext(fixed_seed):
        x = np.sort(np.random.uniform(0, 10, n1))
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.random.uniform(0, 10, n2)
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 101
    result = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=False, assume_x2_is_sorted=False,
            add_subgrid_noise=False)
    result2 = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=True, assume_x2_is_sorted=False,
            add_subgrid_noise=False)
    assert np.allclose(result, result2)


def test_initial_sorting3():
    """
    """
    n1, n2 = int(2e3), int(1e3)

    with NumpyRNGContext(fixed_seed):
        x = np.random.uniform(0, 10, n1)
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.sort(np.random.uniform(0, 10, n2))
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 101
    result = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=False, assume_x2_is_sorted=True,
            add_subgrid_noise=False)
    result2 = conditional_abunmatch(
        x, y, x2, y2, nwin1, assume_x_is_sorted=False, assume_x2_is_sorted=False,
            add_subgrid_noise=False)
    assert np.allclose(result, result2)


def test_initial_sorting4():
    """
    """
    n1, n2 = int(2e3), int(1e3)

    with NumpyRNGContext(fixed_seed):
        x = np.random.uniform(0, 10, n1)
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.random.uniform(0, 10, n2)
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 101
    result = conditional_abunmatch(
        x, y, x2, y2, nwin1,
        assume_x_is_sorted=False, assume_x2_is_sorted=False,
        add_subgrid_noise=False)

    idx_x_sorted = np.argsort(x)
    x_sorted = x[idx_x_sorted]
    y_sorted = y[idx_x_sorted]
    result2 = conditional_abunmatch(
        x_sorted, y_sorted, x2, y2, nwin1,
        assume_x_is_sorted=True, assume_x2_is_sorted=False,
        add_subgrid_noise=False)
    assert np.allclose(result, result2[unsorting_indices(idx_x_sorted)])

    idx_x2_sorted = np.argsort(x2)
    x2_sorted = x2[idx_x2_sorted]
    y2_sorted = y2[idx_x2_sorted]
    result3 = conditional_abunmatch(
        x, y, x2_sorted, y2_sorted, nwin1,
        assume_x_is_sorted=False, assume_x2_is_sorted=True,
        add_subgrid_noise=False)
    assert np.allclose(result, result3)

    result4 = conditional_abunmatch(
        x_sorted, y_sorted, x2_sorted, y2_sorted, nwin1,
        assume_x_is_sorted=True, assume_x2_is_sorted=True,
        add_subgrid_noise=False)
    assert np.allclose(result, result4[unsorting_indices(idx_x_sorted)])


def test_no_subgrid_noise_with_return_indexes():
    """ Enforce that add_subgrid_noise is automatically set to False when return_indexes is True
    """
    n1, n2 = int(1e3), int(1e4)
    x, y = np.arange(n1), np.arange(n1)
    x2, y2 = np.arange(n2), np.arange(n2)
    nwin = 35
    conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=True, return_indexes=True)


def test_return_indexes():
    n1, n2 = int(1e2), int(1e2)

    with NumpyRNGContext(fixed_seed):
        x = np.random.uniform(0, 10, n1)
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed2):
        x2 = np.random.uniform(0, 10, n2)
        y2 = np.random.uniform(-4, -3, n2)

    nwin = 9
    for sorted_x in [False, True]:
        for sorted_x2 in [False, True]:
            x_, y_, x2_, y2_ = x, y, x2, y2
            if sorted_x:
                x_, y_ = np.sort(x_), np.sort(y_)
            if sorted_x2:
                x2_, y2_ = np.sort(x2_), np.sort(y2_)


            values = conditional_abunmatch(x_, y_, x2_, y2_, nwin, add_subgrid_noise=False,
                    assume_x_is_sorted=sorted_x, assume_x2_is_sorted=sorted_x2, return_indexes=False)
            indexes = conditional_abunmatch(x_, y_, x2_, y2_, nwin, add_subgrid_noise=False,
                    assume_x_is_sorted=sorted_x, assume_x2_is_sorted=sorted_x2, return_indexes=True)

            assert np.all(y2_[indexes] == values), "{}, {}".format(sorted_x, sorted_x2)
