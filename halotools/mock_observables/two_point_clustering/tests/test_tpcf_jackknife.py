""" Module providing unit-testing for the `~halotools.mock_observables.tpcf_jackknife` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..tpcf_jackknife import tpcf_jackknife
from ..tpcf import tpcf

slow = pytest.mark.slow

__all__ = ['test_tpcf_jackknife_corr_func', 'test_tpcf_jackknife_cov_matrix']

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rbins = np.linspace(0.001, 0.3, 5).astype(float)
rmax = rbins.max()

fixed_seed = 43


@pytest.mark.slow
def test_tpcf_jackknife_corr_func():
    """
    test the correlation function
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*10, 3))

    result_1, err = tpcf_jackknife(sample1, randoms, rbins,
        Nsub=5, period=period, num_threads=1)

    result_2 = tpcf(sample1, rbins,
        randoms=randoms, period=period, num_threads=1)

    assert np.allclose(result_1, result_2, rtol=1e-09), "correlation functions do not match"


@pytest.mark.slow
def test_tpcf_jackknife_no_pbc():
    """
    test the correlation function
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*10, 3))

    result_1, err = tpcf_jackknife(sample1, randoms, rbins,
        Nsub=5, num_threads=1)


@pytest.mark.slow
def test_tpcf_jackknife_cross_corr():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = np.random.random((Nran*10, 3))

    result = tpcf_jackknife(sample1, randoms, rbins,
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_tpcf_jackknife_no_randoms():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = tpcf_jackknife(sample1, randoms, rbins,
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_tpcf_jackknife_alt_estimator():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = tpcf_jackknife(sample1, randoms, rbins, estimator='Hewett',
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_tpcf_jackknife_cov_matrix():
    """
    test the covariance matrix
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*10, 3))

    nbins = len(rbins)-1

    result_1, err = tpcf_jackknife(sample1, randoms, rbins, Nsub=5, period=period, num_threads=1)

    assert np.shape(err) == (nbins, nbins), "cov matrix not correct shape"
