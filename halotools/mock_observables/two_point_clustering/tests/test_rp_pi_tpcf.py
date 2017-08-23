""" Module provides unit-testing for the `~halotools.mock_observables.rp_pi_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..rp_pi_tpcf import rp_pi_tpcf

__all__ = ('test_rp_pi_tpcf_auto_nonperiodic', 'test_rp_pi_tpcf_auto_periodic',
    'test_rp_pi_tpcf_cross_periodic', 'test_rp_pi_tpcf_cross_nonperiodic')

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rp_bins = np.linspace(0.001, 0.3, 5)
pi_bins = np.linspace(0, 0.3, 5)

fixed_seed = 43


def test_rp_pi_tpcf_auto_nonperiodic():
    """
    test rp_pi_tpcf autocorrelation without periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts, 3))

    result = rp_pi_tpcf(sample1, rp_bins, pi_bins, sample2=None,
        randoms=randoms, period=None, estimator='Natural')

    assert result.ndim == 2, "More than one correlation function returned erroneously."


def test_rp_pi_tpcf_auto_periodic():
    """
    test rp_pi_tpcf autocorrelation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    result = rp_pi_tpcf(sample1, rp_bins, pi_bins, sample2=None,
        randoms=None, period=period, estimator='Natural')

    assert result.ndim == 2, "More than one correlation function returned erroneously."


def test_rp_pi_tpcf_cross_periodic():
    """
    test rp_pi_tpcf cross-correlation without periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    result = rp_pi_tpcf(sample1, rp_bins, pi_bins, sample2=sample2,
        randoms=None, period=period, estimator='Natural')

    assert len(result) == 3, "wrong number of correlations returned"
    assert result[0].ndim == 2, "dimension of auto incorrect"
    assert result[1].ndim == 2, "dimension of cross incorrect"
    assert result[2].ndim == 2, "dimension auto incorrect"


def test_rp_pi_tpcf_cross_nonperiodic():
    """
    test rp_pi_tpcf cross-correlation without periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts, 3))

    result = rp_pi_tpcf(sample1, rp_bins, pi_bins, sample2=sample2,
        randoms=randoms, period=None, estimator='Natural')

    assert len(result) == 3, "wrong number of correlations returned"
    assert result[0].ndim == 2, "dimension of auto incorrect"
    assert result[1].ndim == 2, "dimension of cross incorrect"
    assert result[2].ndim == 2, "dimension auto incorrect"


def test_rp_pi_auto_consistency():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    result_11a, result_12a, result_22a = rp_pi_tpcf(
        sample1, rp_bins, pi_bins, sample2=sample2, period=1,
        do_auto=True, do_cross=True)

    result_11b, result_22b = rp_pi_tpcf(
        sample1, rp_bins, pi_bins, sample2=sample2, period=1,
        do_auto=True, do_cross=False)

    assert np.allclose(result_11a, result_11b)
    assert np.allclose(result_22a, result_22b)


def test_rp_pi_cross_consistency():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    result_11a, result_12a, result_22a = rp_pi_tpcf(
        sample1, rp_bins, pi_bins, sample2=sample2, period=1,
        do_auto=True, do_cross=True)

    result_12b = rp_pi_tpcf(
        sample1, rp_bins, pi_bins, sample2=sample2, period=1,
        do_auto=False, do_cross=True)

    assert np.allclose(result_12a, result_12b)
