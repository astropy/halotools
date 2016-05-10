""" Helper functions used to process arguments passed to functions in 
the isolation_functions sub-package.
"""
import numpy as np

__all__ = ('_get_r_max', '_set_isolation_approx_cell_sizes')

def _get_r_max(sample1, r_max):
    """ Helper function process the input ``r_max`` value and returns 
    the appropriate array after requiring the input is the appropriate 
    size and verifying that all entries are bounded positive numbers. 
    """
    N1 = len(sample1)
    r_max = np.atleast_1d(r_max).astype(float)

    if len(r_max) == 1:
        r_max = np.array([r_max[0]]*N1)
    else:
        try:
            assert len(r_max) == N1
        except AssertionError:
            msg = "Input ``r_max`` must be the same length as ``sample1``."
            raise ValueError(msg)

    try:
        assert np.all(r_max < np.inf)
        assert np.all(r_max > 0)
    except AssertionError:
        msg = "Input ``r_max`` must be an array of bounded positive numbers."
        raise ValueError(msg)

    return r_max

def _set_isolation_approx_cell_sizes(approx_cell1_size, approx_cell2_size, 
    xsearch_length, ysearch_length, zsearch_length):
    """
    """
    if approx_cell1_size is None:
        approx_cell1_size = np.array([xsearch_length, ysearch_length, zsearch_length]).astype(float)
    else:
        approx_cell1_size = np.atleast_1d(approx_cell1_size)
        if len(approx_cell1_size) == 1:
            approx_cell1_size = np.array(
                [approx_cell1_size[0], approx_cell1_size[0], approx_cell1_size[0]]).astype(float)

    try:
        assert approx_cell1_size.shape == (3, )
    except:
        msg = ("Input ``approx_cell1_size`` must be a scalar or length-3 sequence.\n")
        raise ValueError(msg)

    if approx_cell2_size is None:
        approx_cell2_size = np.array([xsearch_length, ysearch_length, zsearch_length]).astype(float)
    else:
        approx_cell2_size = np.atleast_1d(approx_cell2_size)
        if len(approx_cell2_size) == 1:
            approx_cell2_size = np.array(
                [approx_cell2_size[0], approx_cell2_size[0], approx_cell2_size[0]]).astype(float)

    try:
        assert approx_cell2_size.shape == (3, )
    except:
        msg = ("Input ``approx_cell2_size`` must be a scalar or length-3 sequence.\n")
        raise ValueError(msg)

    return approx_cell1_size, approx_cell2_size






