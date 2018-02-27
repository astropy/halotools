"""
Module providing unit-testing for the `~halotools.mock_observables.alignments.w_gplus` function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..ii_plus_projected import ii_plus_projected

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_wplus_returned_shape', 'test_w_plus_threading', 'test_orientation_usage', 'test_round_result', 'test_position_usage')

fixed_seed = 43


def test_w_plus_returned_shape():
    """
    make sure the result that is returned has the correct shape
    """

    ND = 100
    NR = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        randoms = np.random.random((NR, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rp_bins)-1, )

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1, 
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=3)

    assert np.shape(result_2) == (len(rp_bins)-1, )

    # real randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rp_bins)-1, )

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=3)

    assert np.shape(result_2) == (len(rp_bins)-1, )


def test_w_plus_threading():
    """
    test to make sure the results are consistent when num_threads=1 or >1
    """

    ND = 100
    NR = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        randoms = np.random.random((NR, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1)

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)

    # real randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=1)

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)


def test_orientation_usage():
    """
    test to make sure the results are sensitive to the orientations passed in
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation_1 = np.random.random((len(sample1), 2))
    random_orientation_2 = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation_1, random_ellipticities, sample1,
        random_orientation_1, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')

    result_2 = ii_plus_projected(sample1, random_orientation_2, random_ellipticities, sample1,
        random_orientation_2, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')
    
    assert not np.allclose(result_1, result_2)


def test_ellipticity_usage():
    """
    test to make sure the results are sensitive to the ellipticities passed in
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities_1 = np.random.random((len(sample1)))
    random_ellipticities_2 = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities_1, sample1,
        random_orientation, random_ellipticities_1, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities_2, sample1,
        random_orientation, random_ellipticities_2, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')
    
    assert not np.allclose(result_1, result_2)


def test_rpbinning_usage():
    """
    test to make sure the results are sensitive to the rp bin edges passed in
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins_1 = np.linspace(0.001, 0.3, 5)
    rp_bins_2 = np.linspace(0.001, 0.1, 5)

    pi_max = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins_1, pi_max, period=period, num_threads=1, estimator='Natural')

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins_2, pi_max, period=period, num_threads=1, estimator='Natural')
    
    assert not np.allclose(result_1, result_2)


def test_integration_range():
    """
    test to make sure the results are sensitive to the line of sight integration range
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max_1 = 0.05
    pi_max_2 = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max_1, period=period, num_threads=1, estimator='Natural')

    result_2 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max_2, period=period, num_threads=1, estimator='Natural')
    
    assert not np.allclose(result_1, result_2)

def test_round_result():
    """
    test to make sure the projected correlation comes out as zero in the case of non-elliptical input
    """

    ND = 100
    NR = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        sample2 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation_1 = np.random.random((len(sample1), 2))
    random_orientation_2 = np.random.random((len(sample1), 2))
    zero_ellipticities = np.zeros((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation_1, zero_ellipticities, sample2,
        random_orientation_2, zero_ellipticities, rp_bins, pi_max, period=period, num_threads=1)
    
    assert np.allclose(result_1, 0.0)


def test_position_usage():
    """
    test to make sure the results are sensitive to the position coordinates passed in
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        sample2 = np.random.random((ND, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = ii_plus_projected(sample1, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')

    result_2 = ii_plus_projected(sample2, random_orientation, random_ellipticities, sample1,
        random_orientation, random_ellipticities, rp_bins, pi_max, period=period, num_threads=1, estimator='Natural')
    
    assert not np.allclose(result_1, result_2)


