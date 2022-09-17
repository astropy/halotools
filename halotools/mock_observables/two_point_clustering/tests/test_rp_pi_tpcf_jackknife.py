""" Module providing unit-testing for the `~halotools.mock_observables.rp_pi_tpcf_jackknife` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..rp_pi_tpcf_jackknife import rp_pi_tpcf_jackknife
from ..wp_jackknife import wp_jackknife
from ..rp_pi_tpcf import rp_pi_tpcf


slow = pytest.mark.slow

__all__ = ("test_tpcf_jackknife_corr_func",)

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rp_bins = np.linspace(0.001, 0.2, 5).astype(float)
pi_bins = np.linspace(0.0, 0.2, 5)

Npts = 1000

fixed_seed = 43


def test_tpcf_jackknife_corr_func():
    """
    Verify rp_pi_tpcf_jackknife returns the same value as rp_pi_tpcf, PBC case
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 10, 3))

    result_1, err = rp_pi_tpcf_jackknife(
        sample1, randoms, rp_bins, pi_bins, Nsub=3, period=period, num_threads=1
    )

    result_2 = rp_pi_tpcf(
        sample1, rp_bins, pi_bins, randoms=randoms, period=period, num_threads=1
    )

    assert np.allclose(
        result_1, result_2, rtol=1e-09
    ), "correlation functions do not match"


def test_rp_pi_tpcf_jackknife_no_pbc():
    """
    Verify rp_pi_tpcf_jackknife returns the same value as rp_pi_tpcf, no PBC case
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 4, 3))

    result_1, err = rp_pi_tpcf_jackknife(
        sample1, randoms, rp_bins, pi_bins, Nsub=4, num_threads=1
    )

    result_2 = rp_pi_tpcf(sample1, rp_bins, pi_bins, randoms=randoms, num_threads=1)

    assert np.allclose(
        result_1, result_2, rtol=1e-09
    ), "correlation functions do not match"


def test_rp_pi_tpcf_jackknife_cross_corr():
    """
    Verify rp_pi_tpcf_jackknife executes for the case of a cross-correlation
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = np.random.random((Nran, 3))

    result = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        period=period,
        Nsub=3,
        num_threads=1,
        sample2=sample2,
    )


def test_rp_pi_tpcf_jackknife_no_randoms():
    """
    Verify the correlation function executes when passed in [Nran] for randoms
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        period=period,
        Nsub=3,
        num_threads=1,
        sample2=sample2,
    )


def test_rp_pi_tpcf_jackknife_alt_estimator():
    """
    Verify the correlation function executes for the ``Landy-Szalay`` estimator
    """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        estimator="Landy-Szalay",
        period=period,
        Nsub=3,
        num_threads=1,
        sample2=sample2,
    )


def test_rp_pi_tpcf_jackknife_cov_matrix():
    """
    Verify the covariance matrix returned by rp_pi_tpcf_jackknife has the correct shape
    """
    Npts = 300
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 3, 3))

    nbins_1 = len(rp_bins) - 1
    nbins_2 = len(pi_bins) - 1

    result_1, err = rp_pi_tpcf_jackknife(
        sample1, randoms, rp_bins, pi_bins, Nsub=3, period=period, num_threads=1
    )

    assert np.shape(err) == (
        nbins_1 * nbins_2,
        nbins_1 * nbins_2,
    ), "cov matrix not correct shape"


def test_do_auto_false():
    """ """
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = [Nran]

    result1 = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        period=period,
        Nsub=3,
        num_threads=1,
        sample2=sample2,
        do_auto=False,
    )


def test_consistency_with_wp_jackknife():
    """
    Verify the result and covariance matrix is consistent with wp_jackknife
    """

    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 3, 3))

    # use one large pi_bin
    pi_max = 0.1
    pi_min = 0.0
    pi_bins_this_test = np.array([pi_min, pi_max])

    result_1, err_1 = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins_this_test,
        Nsub=3,
        period=period,
        num_threads=1,
    )

    result_2, err_2 = wp_jackknife(
        sample1, randoms, rp_bins, pi_max=pi_max, Nsub=3, period=period, num_threads=1
    )

    # account for wp integration
    result_1 = result_1.flatten() * 2.0 * np.diff(pi_bins_this_test)

    assert np.all(
        result_1 == result_2
    ), "tpcf is not consistent between rp_pi_tpcf and wp"

    # account for wp integration
    factor = (2.0 * np.diff(pi_bins_this_test)) ** 2.0
    err_1 = err_1 * factor[0]

    assert np.allclose(err_1, err_2), "cov matrix is not consoistent with wp_jackknife"


def test_consistency_with_wp_jackknife_2():
    """
    Verify the result and covariance matrix is consistent with wp_jackknife when len(pi_bins)>2
    """

    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts * 3, 3))

    # use one large pi_bin
    pi_max = 0.15
    pi_min = 0.0
    pi_bins_this_test = np.linspace(pi_min, pi_max, 4)

    __, cov1 = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins_this_test,
        Nsub=3,
        period=period,
        num_threads=1,
    )

    Nrp_bins = len(rp_bins) - 1
    Npi_bins = len(pi_bins_this_test) - 1
    print(Nrp_bins, Npi_bins)
    assert np.shape(cov1) == (
        Nrp_bins * Npi_bins,
        Nrp_bins * Npi_bins,
    ), "covariance matrix is not the correct shape"

    __, cov2 = wp_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_max=pi_bins_this_test[1],
        Nsub=3,
        period=period,
        num_threads=1,
    )

    # account for wp integration in comparison
    factor = (2.0 * np.diff(pi_bins_this_test[0:2])) ** 2.0
    cov1 = cov1 * factor[0]

    j, l = 0, 0  # first pi_bins
    for i in range(0, Nrp_bins):
        for k in range(0, Nrp_bins):
            ind_1 = Npi_bins * i + j
            ind_2 = Npi_bins * k + l
            element_1 = cov1[ind_1, ind_2]
            element_2 = cov2[i, k]
            assert np.allclose(element_1, element_2), "covariance elements do no match"


def test_parallel_serial_consistency_1():
    Npts1, Nran = 300, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        randoms = np.random.random((Nran, 3))

    xi1, cov1 = rp_pi_tpcf_jackknife(
        sample1, randoms, rp_bins, pi_bins, period=period, Nsub=3, num_threads=1
    )
    xi2, cov2 = rp_pi_tpcf_jackknife(
        sample1, randoms, rp_bins, pi_bins, period=period, Nsub=3, num_threads=2
    )

    assert np.allclose(
        xi1, xi2
    ), "tpcf between threaded and non-threaded results do no match"
    assert np.allclose(
        cov1, cov2
    ), "cov matrix between threaded and non-threaded results do no match"


def test_parallel_serial_consistency_2():
    Npts1, Npts2, Nran = 300, 180, 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))
        randoms = np.random.random((Nran, 3))

    xi1, cov1 = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        period=period,
        Nsub=3,
        num_threads=1,
        sample2=sample2,
        do_auto=False,
    )

    xi2, cov2 = rp_pi_tpcf_jackknife(
        sample1,
        randoms,
        rp_bins,
        pi_bins,
        period=period,
        Nsub=3,
        num_threads=2,
        sample2=sample2,
        do_auto=False,
    )

    assert np.allclose(
        xi1, xi2
    ), "tpcf between threaded and non-threaded results do no match"
    assert np.allclose(
        cov1, cov2
    ), "cov matrix between threaded and non-threaded results do no match"
