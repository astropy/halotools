""" Module provides unit-testing for the `~halotools.mock_observables.s_mu_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from ..s_mu_tpcf import s_mu_tpcf

__all__ = ['test_s_mu_tpcf_auto_periodic', 'test_s_mu_tpcf_auto_nonperiodic']

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
    mu_bins = np.linspace(0, 1.0, 5)

    result_1 = s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None,
        randoms=randoms, period=None, estimator='Natural')

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
        randoms=None, period=period, estimator='Natural')

    assert result_1.ndim == 2, "correlation function returned has wrong dimension."


def test_s_mu_cross_consistency():
    """
    """
    Npts1, Npts2 = 100, 200

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
    period = np.array([1.0, 1.0, 1.0])
    s_bins = np.linspace(0.001, 0.3, 5)
    mu_bins = np.linspace(0, 1.0, 10)

    result_11a, result_12a, result_22a = s_mu_tpcf(
        sample1, s_bins, mu_bins, sample2=sample2,
        period=period, do_auto=True, do_cross=True)

    result_12b = s_mu_tpcf(
        sample1, s_bins, mu_bins, sample2=sample2,
        period=period, do_auto=False, do_cross=True)

    assert np.allclose(result_12a, result_12b)


def test_s_mu_auto_consistency():
    """
    """
    Npts1, Npts2 = 100, 200

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
    period = np.array([1.0, 1.0, 1.0])
    s_bins = np.linspace(0.001, 0.3, 5)
    mu_bins = np.linspace(0, 1.0, 10)

    result_11a, result_12a, result_22a = s_mu_tpcf(
        sample1, s_bins, mu_bins, sample2=sample2,
        period=period, do_auto=True, do_cross=True)

    result_11b, result_22b = s_mu_tpcf(
        sample1, s_bins, mu_bins, sample2=sample2,
        period=period, do_auto=True, do_cross=False)

    assert np.allclose(result_11a, result_11b)
    assert np.allclose(result_22a, result_22b)
