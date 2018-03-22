"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ....utils.conditional_percentile import cython_sliding_rank
from .naive_python_cam import sample2_window_indices, pure_python_rank_matching

fixed_seed = 43


def test_pure_python1():
    """
    """
    n1, n2, nwin = 5001, 1001, 11
    nhalfwin = nwin/2
    x_sample1 = np.linspace(0, 1, n1)
    with NumpyRNGContext(fixed_seed):
        y_sample1 = np.random.uniform(0, 1, n1)
    ranks_sample1 = cython_sliding_rank(x_sample1, y_sample1, nwin)

    x_sample2 = np.linspace(0, 1, n2)
    with NumpyRNGContext(fixed_seed):
        y_sample2 = np.random.uniform(-4, -3, n2)
    ranks_sample2 = cython_sliding_rank(x_sample2, y_sample2, nwin)

    result = pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin)

    for ix1 in range(2*nwin, n1-2*nwin):

        rank1 = ranks_sample1[ix1]
        low, high = sample2_window_indices(ix1, x_sample1, x_sample2, nwin)

        sorted_window2 = np.sort(y_sample2[low:high])
        assert len(sorted_window2) == nwin

        correct_result_ix1 = sorted_window2[rank1]

        assert correct_result_ix1 == result[ix1]


def test_hard_coded_case1():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.03, 0.54, 0.67, 0.73, 0.86]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case2():
    """
    """
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]

    assert np.allclose(pure_python_result, correct_result)


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

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.04, 0.3, 0.6, 5., 10.]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case4():
    """ Every x2 is larger than the largest x.

    So the only CAM window ever used is the first 3 elements of y2.
    """
    nwin = 3

    x = np.array((0., 0., 0., 0., 0.))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.04, 0.3, 0.3, 0.3, 0.6]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case5():
    """ Every x2 is smaller than the smallest x.

    So the only CAM window ever used is the final 3 elements of y2.
    """
    nwin = 3

    x = np.array((1., 1., 1, 1, 1))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.6, 5, 5, 5, 10]

    assert np.allclose(pure_python_result, correct_result)
