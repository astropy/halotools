""" Helper functions used to process arguments passed to functions in
the isolation_functions sub-package.
"""
import numpy as np

from ...custom_exceptions import HalotoolsError

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


def _func_signature_int_from_cond_func(cond_func):
    """
    Return the number of marks-per-point expected by the
    marking function chosen by the input ``cond_func``.
    """

    try:
        assert int(cond_func) == cond_func
    except:
        msg = ("Your input ``cond_func`` parameter must be one of the integer values \n"
        "associated with one of the marking functions for conditional isolation.\n"
        "See the docstring of either `~halotools.mock_observables.conditional_spherical_isolation`\n"
        "or `~halotools.mock_observables.conditional_spherical_isolation` for a list of available options.\n")
        raise ValueError(msg)

    if cond_func == 0:
        return 1
    elif cond_func == 1:
        return 1
    elif cond_func == 2:
        return 1
    elif cond_func == 3:
        return 1
    elif cond_func == 4:
        return 1
    elif cond_func == 5:
        return 2
    elif cond_func == 6:
        return 2
    else:
        msg = ("Your input ``cond_func`` = %i parameter must be one of the integer values \n"
        "associated with one of the marking functions for conditional isolation.\n"
        "See the docstring of either `~halotools.mock_observables.conditional_spherical_isolation`\n"
        "or `~halotools.mock_observables.conditional_spherical_isolation` for a list of available options.\n")
        raise HalotoolsError(msg % cond_func)


def reshape_input_marks(marks, npts_sample, correct_num_marks, cond_func):
    """
    """
    if marks is None:
        marks = np.ones((npts_sample, correct_num_marks), dtype=np.float64)
    else:
        marks = np.atleast_1d(marks).astype(np.float64)
        if len(marks.shape) == 1:
            marks = np.reshape(marks, (len(marks), 1))

    try:
        npts_in_marks = marks.shape[0]
        assert npts_in_marks == npts_sample
    except AssertionError:
        msg = ("You must pass in an array of marks whose length \n"
            "matches the number of points in your sample.\n"
            "Your input sample has %i points, but your marks array has length = %i.\n")
        raise ValueError(msg % (npts_sample, npts_in_marks))

    try:
        num_input_marks_per_point = marks.shape[1]
        assert num_input_marks_per_point == correct_num_marks
    except AssertionError:
        msg = ("The input value of ``cond_func`` = %i. "
            "For this choice for the conditional isolation function,\n"
            "there should be %i marks per point. \n"
            "However, your input marks has %i points per mark.\n")
        raise ValueError(msg % (cond_func, correct_num_marks, num_input_marks_per_point))

    return marks


def _conditional_isolation_process_marks(sample1, sample2, marks1, marks2, cond_func):
    """
    private function to process the arguments for conditional isolation functions
    """

    correct_num_marks = _func_signature_int_from_cond_func(cond_func)
    npts_sample1 = sample1.shape[0]
    npts_sample2 = sample2.shape[0]

    marks1 = reshape_input_marks(marks1, npts_sample1, correct_num_marks, cond_func)
    marks2 = reshape_input_marks(marks2, npts_sample2, correct_num_marks, cond_func)

    return marks1, marks2
