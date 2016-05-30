""" Module provides unit-testing for the `~halotools.mock_observables.s_mu_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..s_mu_tpcf import s_mu_tpcf

__all__=['test_s_mu_tpcf_auto_periodic', 'test_s_mu_tpcf_auto_nonperiodic']

fixed_seed = 43


def test_s_mu_tpcf_auto_nonperiodic():
    """
    test s_mu_tpcf autocorrelation without periodic boundary conditons.
    """
    Npts, Nran = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Nran, 3))
    s_bins = np.linspace(0.001, 0.3, 5)
    mu_bins = np.linspace(0, 1.0, 10)

    result_1 = s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None,
        randoms=randoms, period=None,
        max_sample_size=int(1e4), estimator='Natural')

    assert result_1.ndim == 2, "correlation function returned has wrong dimension."


def test_s_mu_tpcf_auto_periodic():
    """
    test s_mu_tpcf autocorrelation with periodic boundary conditons.
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    s_bins = np.linspace(0.001, 0.3, 5)
    mu_bins = np.linspace(0, 1.0, 10)

    result_1 = s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None,
        randoms=None, period=period,
        max_sample_size=int(1e4), estimator='Natural')

    assert result_1.ndim == 2, "correlation function returned has wrong dimension."
