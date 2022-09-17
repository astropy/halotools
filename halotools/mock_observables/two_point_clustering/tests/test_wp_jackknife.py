""" Module providing unit-testing for the `~halotools.mock_observables.tpcf_jackknife` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..wp_jackknife import wp_jackknife
from ..wp import wp
from ...tests.cf_helpers import generate_3d_regular_mesh

slow = pytest.mark.slow

__all__ = ["test_tpcf_jackknife_corr_func", "test_wp_jackknife_cov_matrix"]

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rp_bins = np.linspace(0.001, 0.2, 5).astype(float)
rp_max = rp_bins.max()
pi_max = 0.3

fixed_seed = 43


def test_tpcf_jackknife_corr_func():
    """
    test the correlation function
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 10, 3))

    randoms = np.concatenate(
        (randoms, generate_3d_regular_mesh(20, dmin=0, dmax=period))
    )

    result_1, err = wp_jackknife(
        sample1, randoms, rp_bins, pi_max, Nsub=2, period=period, num_threads=1
    )

    result_2 = wp(
        sample1, rp_bins, pi_max, randoms=randoms, period=period, num_threads=1
    )

    assert np.allclose(
        result_1, result_2, rtol=1e-09
    ), "correlation functions do not match"


def test_wp_jackknife_no_pbc():
    """
    test the correlation function
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 10, 3))

    result_1, err = wp_jackknife(
        sample1, randoms, rp_bins, pi_max, Nsub=2, num_threads=1
    )


def test_wp_jackknife_cross_corr():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = np.random.random((Nran * 10, 3))

    result = wp_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_max,
        period=period,
        Nsub=2,
        num_threads=1,
        sample2=sample2,
    )


def test_wp_jackknife_no_randoms():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = wp_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_max,
        period=period,
        Nsub=2,
        num_threads=1,
        sample2=sample2,
    )


def test_wp_jackknife_alt_estimator():
    """
    test the correlation function
    """
    Npts1, Npts2, Nran = 100, 90, 500
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = wp_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_max,
        estimator="Landy-Szalay",
        period=period,
        Nsub=2,
        num_threads=1,
        sample2=sample2,
    )


def test_wp_jackknife_cov_matrix():
    """
    test the covariance matrix
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 10, 3))

    nbins = len(rp_bins) - 1

    result_1, err = wp_jackknife(
        sample1, randoms, rp_bins, pi_max, Nsub=2, period=period, num_threads=1
    )

    assert np.shape(err) == (nbins, nbins), "cov matrix not correct shape"


def test_do_auto_false():
    """ """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    # result1 = wp_jackknife(sample1, randoms, rp_bins, pi_max,
    #     period=period, Nsub=3, num_threads=1, sample2=sample2,
    #     do_auto=False)
    result = wp_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_max,
        period=period,
        Nsub=2,
        num_threads=1,
        sample2=sample2,
        do_auto=False,
    )
