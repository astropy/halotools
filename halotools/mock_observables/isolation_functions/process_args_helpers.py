""" Helper functions used to process arguments passed to functions in 
the isolation_functions sub-package.
"""
import numpy as np
from warnings import warn
import multiprocessing
num_available_cores = multiprocessing.cpu_count()

__all__ = ('_get_num_threads', '_get_r_max', '_get_period', 
    '_set_spherical_isolation_approx_cell_sizes')

def _get_num_threads(input_num_threads, enforce_max_cores = False):
    """ Helper function requires that ``input_num_threads`` either be an 
    integer or the string ``max``. If ``input_num_threads`` exceeds the 
    number of available cores, a warning will be issued. 
    In this event,  ``enforce_max_cores`` is set to True, 
    then ``num_threads`` is automatically set to num_cores. 
    """
    if input_num_threads=='max':
        num_threads = num_available_cores
    else:
        try:
            num_threads = int(input_num_threads)
            assert num_threads == input_num_threads
        except:
            msg = ("Input ``num_threads`` must be an integer")
            raise ValueError(msg)

    if num_threads > num_available_cores:
        msg = ("Input ``num_threads`` = %i exceeds the ``num_available_cores`` = %i.\n")

        if enforce_max_cores is True:
            msg += ("Since ``enforce_max_cores`` is True, "
                "setting ``num_threads`` to ``num_available_cores``.\n")
            num_threads = num_available_cores

        warn(msg % (num_threads, num_available_cores))

    return num_threads

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






