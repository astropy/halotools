""" Helper functions used to process arguments passed to functions in 
the isolation_functions sub-package.
"""
import numpy as np

__all__ = ('_get_r_max', '_get_period', '_set_spherical_isolation_approx_cell_sizes')

def _get_r_max(data1, r_max):
    """ Helper function process the input ``r_max`` value and returns 
    the appropriate array after requiring the input is the appropriate 
    size and verifying that all entries are bounded positive numbers. 
    """
    N1 = len(data1)
    r_max = np.atleast_1d(r_max).astype(float)

    if len(r_max) == 1:
        r_max = np.array([r_max[0]]*N1)

    try:
        assert np.all(r_max < np.inf)
        assert np.all(r_max > 0)
    except AssertionError:
        msg = "Input ``r_max`` must be an array of bounded positive numbers."
        raise ValueError(msg)

    return r_max

def _get_period(period):
    """ Helper function used to process the input ``period`` argument. 
    If ``period`` is set to None, function returns (None, False). 
    Otherwise, function returns a 3-element array.
    """

    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)

        if len(period) == 1:
            period = np.array([period[0]]*3).astype(float)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    return period, PBCs

def _set_spherical_isolation_approx_cell_sizes(approx_cell1_size, approx_cell2_size, max_r_max):
    """
    """
    if approx_cell1_size is None:
        approx_cell1_size = np.array([max_r_max, max_r_max, max_r_max]).astype(float)
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
        approx_cell2_size = np.array([max_r_max, max_r_max, max_r_max]).astype(float)
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






