""" Module providing unit-testing of the `~halotools.mock_observables.wp` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from .locate_external_unit_testing_data import wp_corrfunc_comparison_files_exist
from ..wp import wp

__all__ = ('test_wp_auto_nonperiodic', 'test_wp_auto_periodic', 'test_wp_cross_periodic',
    'test_wp_cross_nonperiodic')

period = np.array([1.0, 1.0, 1.0])
rp_bins = np.linspace(0.001, 0.3, 3)
pi_max = 0.3

fixed_seed = 43

WP_CORRFUNC_FILES_EXIST = wp_corrfunc_comparison_files_exist()


def test_wp_auto_nonperiodic():
    """
    test wp autocorrelation without periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts, 3))

    result = wp(sample1, rp_bins, pi_max, sample2=None,
                randoms=randoms, period=None, estimator='Natural')

    print(result)
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_wp_auto_periodic():
    """
    test wp autocorrelation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    result = wp(sample1, rp_bins, pi_max, sample2=None,
                randoms=None, period=period, estimator='Natural')

    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_wp_cross_periodic():
    """
    test wp cross-correlation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    result = wp(sample1, rp_bins, pi_max, sample2=sample2,
                randoms=None, period=period, estimator='Natural')

    assert len(result) == 3, "wrong number of correlations returned"
    assert result[0].ndim == 1, "dimension of auto incorrect"
    assert result[1].ndim == 1, "dimension of cross incorrect"
    assert result[2].ndim == 1, "dimension auto incorrect"


def test_wp_cross_nonperiodic():
    """
    test wp cross-correlation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))
        randoms = np.random.random((Npts, 3))

    result = wp(sample1, rp_bins, pi_max, sample2=sample2,
                randoms=randoms, period=None, estimator='Natural')

    assert len(result) == 3, "wrong number of correlations returned"
    assert result[0].ndim == 1, "dimension of auto incorrect"
    assert result[1].ndim == 1, "dimension of cross incorrect"
    assert result[2].ndim == 1, "dimension auto incorrect"


@pytest.mark.skipif('not WP_CORRFUNC_FILES_EXIST')
def test_wp_vs_corrfunc():
    """
    """
    msg = ("This unit-test compares the wp results from halotools \n"
        "against the results derived from the Corrfunc code managed by \n"
        "Manodeep Sinha. ")
    __, aph_fname1, aph_fname2, aph_fname3, deep_fname1, deep_fname2 = (
        wp_corrfunc_comparison_files_exist(return_fnames=True))

    sinha_sample1_wp = np.load(deep_fname1)[:, 0]
    sinha_sample2_wp = np.load(deep_fname2)[:, 0]

    sample1 = np.load(aph_fname1)
    sample2 = np.load(aph_fname2)
    rp_bins = np.load(aph_fname3)
    pi_max = 40.0

    halotools_result1 = wp(sample1, rp_bins, pi_max, period=250.0)
    assert np.allclose(halotools_result1, sinha_sample1_wp, rtol=1e-3), msg

    halotools_result2 = wp(sample2, rp_bins, pi_max, period=250.0)
    assert np.allclose(halotools_result2, sinha_sample2_wp, rtol=1e-3), msg


def test_wp_cross_consistency():
    Npts1, Npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    wp_11a, wp_12a, wp_22a = wp(sample1, rp_bins, pi_max, sample2=sample2, period=1,
        do_auto=True, do_cross=True)
    assert np.shape(wp_11a) == (len(rp_bins)-1, )
    assert np.shape(wp_12a) == (len(rp_bins)-1, )
    assert np.shape(wp_22a) == (len(rp_bins)-1, )

    wp_12b = wp(sample1, rp_bins, pi_max, sample2=sample2, period=1,
        do_auto=False, do_cross=True)
    assert np.shape(wp_12b) == (len(rp_bins)-1, )

    assert np.allclose(wp_12a, wp_12b)


def test_wp_auto_consistency():
    Npts1, Npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    wp_11a, wp_12a, wp_22a = wp(sample1, rp_bins, pi_max, sample2=sample2, period=1,
        do_auto=True, do_cross=True)

    wp_11b, wp_22b = wp(sample1, rp_bins, pi_max, sample2=sample2, period=1,
        do_auto=True, do_cross=False)

    assert np.allclose(wp_11a, wp_11b)
    assert np.allclose(wp_22a, wp_22b)
