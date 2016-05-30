""" Module providing unit-testing of `~halotools.mock_observables.delta_sigma`.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..delta_sigma import delta_sigma

__all__ = ['test_delta_sigma1']

fixed_seed = 43


def test_delta_sigma1():
    """ Simple unit-test of delta_sigma. Does not verify correctness.
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((10000, 3))
        sample2 = np.random.random((10000, 3))
    rp_bins = np.logspace(-2, -1, 5)
    pi_max = 0.1
    ds = delta_sigma(sample1, sample2, rp_bins, pi_max, period=1, log_bins=False)
    assert ds.ndim ==1, 'wrong number of results returned'
