""" Module containing various helper functions used to process the
arguments of functions throughout the `~halotools.mock_observables` package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from warnings import warn
import numpy as np
import multiprocessing

from ..utils.array_utils import array_is_monotonic


num_available_cores = multiprocessing.cpu_count()

__all__ = ('enforce_sample_respects_pbcs', 'get_num_threads', 'get_period',
    'enforce_sample_has_correct_shape', 'get_separation_bins_array')


def enforce_sample_respects_pbcs(x, y, z, period):
    """ Verify that the input sample is properly bounded in all dimensions by the input period.

    Parameters
    -----------
    x, y, z : arrays

    period : 3-element sequence
    """
    try:
        assert np.all(x >= 0)
        assert np.all(y >= 0)
        assert np.all(z >= 0)
    except:
        msg = ("You set periodic boundary conditions to be True by passing in \n"
            "period = (%.2f, %.2f, %.2f), but your input data has negative values,\n"
            "indicating that you forgot to apply periodic boundary conditions.\n")
        raise ValueError(msg % (period[0], period[1], period[2]))

    try:
        assert np.all(x <= period[0])
    except:
        msg = ("You set xperiod = %.2f but there are values in the x-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[0])

    try:
        assert np.all(y <= period[1])
    except:
        msg = ("You set yperiod = %.2f but there are values in the y-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[1])

    try:
        assert np.all(z <= period[2])
    except:
        msg = ("You set zperiod = %.2f but there are values in the z-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[2])


def get_num_threads(input_num_threads, enforce_max_cores=False):
    """ Helper function requires that ``input_num_threads`` either be an
    integer or the string ``max``. If ``input_num_threads`` exceeds the
    number of available cores, a warning will be issued.
    In this event,  ``enforce_max_cores`` is set to True,
    then ``num_threads`` is automatically set to num_cores.
    """
    if input_num_threads == 'max':
        num_threads = num_available_cores
    else:
        try:
            num_threads = int(input_num_threads)
            assert num_threads == input_num_threads
        except:
            msg = ("Input ``num_threads`` must be an integer")
            raise ValueError(msg)

    if num_threads > num_available_cores:
        msg = ("Input ``num_threads`` = {0} exceeds the ``num_available_cores`` = {1}.\n")
        warn(msg.format(num_threads, num_available_cores))

        if enforce_max_cores is True:
            msg = ("Since ``enforce_max_cores`` is True,\n"
                "setting ``num_threads`` to ``num_available_cores``.")
            warn(msg)

            num_threads = num_available_cores

    return num_threads


def get_period(period):
    """ Helper function used to process the input ``period`` argument.
    If ``period`` is set to None, function returns period, PBCs = (None, False).
    Otherwise, function returns ([period, period, period], True).
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
            assert len(period) == 3
        except AssertionError:
            msg = ("Input ``period`` must be either a scalar or a 3-element sequence.\n"
                "All values must bounded positive numbers.\n")
            raise ValueError(msg)

    return period, PBCs


def enforce_sample_has_correct_shape(sample, ndim=3):
    """ Function inspects the input ``sample`` and enforces that it is of shape (Npts, 3).
    """
    sample = np.atleast_1d(sample)
    try:
        input_shape = np.shape(sample)
        assert len(input_shape) == 2
        assert input_shape[1] == ndim
    except:
        msg = ("Input sample of points must be a Numpy ndarray of shape (Npts, {0}).\n"
            "To convert a sequence of 1d arrays x, y, z into correct shape expected \n"
            "throughout the `mock_observables` package:\n\n"
            ">>> sample = np.vstack([x, y, z]).T ".format(ndim))
        raise TypeError(msg)
    return sample


def get_separation_bins_array(separation_bins):
    """ Function verifies that the input ``separation_bins`` is a monotonically increasing
    1d Numpy array with at least two entries, all of which are required to be strictly positive.

    This helper function can be used equally well with 3d separation bins ``rbins`` or
    2d projected separation bins ``rp_bins``.

    """
    separation_bins = np.atleast_1d(separation_bins)

    try:
        assert separation_bins.ndim == 1
        assert len(separation_bins) > 1
        # cbx_aph: There are lots of places like this where we never check that the array is increasing if it has 2 elements. The reason for this is that array_is_monotonic requires 3 elements. This could easily be fixed because I think we could allow 2 elements arrays to array_is_monotonic - it would always be monotonic but we would know whether it was increasing or decreasing
        if len(separation_bins) > 2:
            assert array_is_monotonic(separation_bins, strict=True) == 1
        assert np.all(separation_bins > 0)
    except AssertionError:
        msg = ("\n Input separation bins must be a monotonically increasing \n"
               "1-D array with at least two entries, all of which must be strictly positive.\n")
        raise TypeError(msg)

    return separation_bins


def get_line_of_sight_bins_array(pi_bins):
    """ Function verifies that the input ``pi_bins`` is a monotonically increasing
    1d Numpy array with at least two entries. The  `get_line_of_sight_bins_array` function differs from
    the `get_separation_bins_array` function only in that values of zero are permissible.

    """
    pi_bins = np.atleast_1d(pi_bins)

    try:
        assert pi_bins.ndim == 1
        assert len(pi_bins) > 1
        if len(pi_bins) > 2:
            assert array_is_monotonic(pi_bins, strict=True) == 1
    except AssertionError:
        msg = ("\n Input separation bins must be a monotonically increasing \n"
               "1-D array with at least two entries.\n")
        raise TypeError(msg)

    return pi_bins
