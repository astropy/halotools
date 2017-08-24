"""
Helper functions used by modules in `~halotools.mock_observables.pairwise_velocities`
to process arguments and raise appropriate exceptions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from warnings import warn
from astropy.utils.misc import NumpyRNGContext

from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_period, get_num_threads)

from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import array_is_monotonic

__all__ = ['_pairwise_velocity_stats_process_args', '_process_radial_bins', '_process_rp_bins']
__author__ = ['Duncan Campbell']


def _pairwise_velocity_stats_process_args(sample1, velocities1, sample2, velocities2,
        period, do_auto, do_cross, num_threads, approx_cell1_size, approx_cell2_size, seed):
    """
    Private method to do bounds-checking on the arguments passed to
    `~halotools.mock_observables.pairwise_velocity_stats`.
    """

    sample1 = enforce_sample_has_correct_shape(sample1)
    velocities1 = np.atleast_1d(velocities1).astype('f4')

    if sample2 is not None:
        sample2 = np.atleast_1d(sample2)
        if velocities2 is None:
            msg = ("\n If `sample2` is passed as an argument, \n"
                   "`velocities2` must also be specified.")
            raise ValueError(msg)
        else:
            velocities2 = np.atleast_1d(velocities2)
        if np.all(sample1 == sample2):
            _sample1_is_sample2 = True
            msg = ("\n Warning: `sample1` and `sample2` are exactly the same, \n"
                   "only the auto-function will be returned.\n")
            warn(msg)
            do_cross = False
        else:
            _sample1_is_sample2 = False
    else:
        sample2 = sample1
        velocities2 = velocities1
        _sample1_is_sample2 = True

    period, PBCs = get_period(period)

    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        msg = ('\n `do_auto` and `do_cross` keywords must be of type boolean.')
        raise HalotoolsError(msg)
    else:
        if (do_auto is False) & (do_cross is False):
            msg = ("Both ``do_auto`` and ``do_cross`` have been set to False")
            raise ValueError(msg)

    num_threads = get_num_threads(num_threads)

    return sample1, velocities1, sample2, velocities2, period, do_auto,\
        do_cross, num_threads, _sample1_is_sample2, PBCs


def _process_radial_bins(rbins, period, PBCs):
    """
    process radial bin parameter
    """

    # check the radial bins parameter
    rbins = np.atleast_1d(rbins)
    rmax = np.max(rbins)
    try:
        assert rbins.ndim == 1
        assert len(rbins) > 1
        if len(rbins) > 2:
            assert array_is_monotonic(rbins, strict=True) == 1
    except AssertionError:
        msg = ("\n Input `rbins` must be a monotonically increasing \n"
               "1-D array with at least two entries.")
        raise ValueError(msg)

    # check for input parameter consistency
    if PBCs:
        if (rmax >= np.min(period)/3.0):
            msg = ("\n The maximum length over which you search for pairs \n"
                   "of points cannot be larger than Lbox/3 in any dimension. \n"
                   "If you need to count pairs on these length scales, \n"
                   "you should use a larger simulation. \n")
            raise ValueError(msg)

    return rbins


def _process_rp_bins(rp_bins, pi_max, period, PBCs):
    """
    process projected radial bin and pi_max parameters
    """

    # process projected radial bins
    rp_bins = np.atleast_1d(rp_bins)
    rp_max = np.max(rp_bins)
    try:
        assert rp_bins.ndim == 1
        assert len(rp_bins) > 1
        if len(rp_bins) > 2:
            assert array_is_monotonic(rp_bins, strict=True) == 1
    except AssertionError:
        msg = ("\n Input `rp_bins` must be a monotonically increasing \n"
               "1-D array with at least two entries.")
        raise ValueError(msg)

    pi_max = float(pi_max)

    if PBCs:
        if (rp_max >= np.min(period[0:2])/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                   "cannot be larger than Lbox/3 in any dimension. \n"
                   "If you need to count pairs on these length scales, \n"
                   "you should use a larger simulation.\n")
            raise ValueError(msg)
        if (pi_max >= np.min(period[2])/3.0):
            msg = ("\n The maximum length over which you search for pairs of points \n"
                   "cannot be larger than Lbox/3 in any dimension. \n"
                   "The input ``pi_max`` = {0}"
                   "If you need to count pairs on these length scales, \n"
                   "you should use a larger simulation.\n".format(pi_max))
            raise ValueError(msg)

    return rp_bins, pi_max
