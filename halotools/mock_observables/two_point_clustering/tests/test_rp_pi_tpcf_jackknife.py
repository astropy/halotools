""" Module providing unit-testing for the `~halotools.mock_observables.rp_pi_tpcf_jackknife` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..rp_pi_tpcf_jackknife import rp_pi_tpcf_jackknife
from ..rp_pi_tpcf import rp_pi_tpcf

slow = pytest.mark.slow

__all__ = ['test_rp_pi_tpcf_jackknife_corr_func', 'test_rp_pi_tpcf_jackknife_cov_matrix']

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rp_bins = np.linspace(0.001, 0.2, 5).astype(float)
pi_bins = np.linspace(0.001, 0.2, 5)

Npts = 1000

fixed_seed = 43


@pytest.mark.slow
def test_tpcf_jackknife_corr_func():
    """
    test the correlation function
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*10, 3))

    result_1, err = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins,
        Nsub=5, period=period, num_threads=1)

    result_2 = rp_pi_tpcf(sample1, rp_bins, pi_bins,
        randoms=randoms, period=period, num_threads=1)

    assert np.allclose(result_1, result_2, rtol=1e-09), "correlation functions do not match"


@pytest.mark.slow
def test_rp_pi_tpcf_jackknife_no_pbc():
    """
    test the correlation function
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*30, 3))

    result_1, err = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins,
        Nsub=5, num_threads=1)


@pytest.mark.slow
def test_rp_pi_tpcf_jackknife_cross_corr():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = np.random.random((Nran*30, 3))

    result = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins,
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_rp_pi_tpcf_jackknife_no_randoms():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins,
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_rp_pi_tpcf_jackknife_alt_estimator():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins, estimator='Hewett',
        period=period, Nsub=3, num_threads=1, sample2=sample2)


@pytest.mark.slow
def test_rp_pi_tpcf_jackknife_cov_matrix():
    """
    test the covariance matrix
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts*30, 3))

    nbins_1 = len(rp_bins)-1
    nbins_2 = len(pi_bins)-1

    result_1, err = rp_pi_tpcf_jackknife(sample1, randoms, rp_bins, pi_bins, Nsub=5, period=period, num_threads=1)

    assert np.shape(err) == (nbins_1*nbins_2, nbins_1*nbins_2), "cov matrix not correct shape"
