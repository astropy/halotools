"""
Module providing unit-testing for the `~halotools.mock_observables.alignments.w_gplus` function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..gi_minus_projected import gi_minus_projected

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_w_gplus_returned_shape', 'test_w_gplus_threading', 'test_orientation_usage')

fixed_seed = 43


def test_w_gminus_returned_shape():
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
    result_1 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rp_bins)-1, )

    result_2 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=3)

    assert np.shape(result_2) == (len(rp_bins)-1, )

    # real randoms
    result_1 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rp_bins)-1, )

    result_2 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=3)

    assert np.shape(result_2) == (len(rp_bins)-1, )


def test_w_gminus_threading():
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
    result_1 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=1)

    result_2 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)

    # real randoms
    result_1 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=1)

    result_2 = gi_minus_projected(sample1, random_orientation, random_ellipticities, sample1,
        rp_bins, pi_max, randoms1=randoms, randoms2=randoms, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)


def test_orientation_usage():
    """
    test to make sure the results are sensitive to the orientations passed in
    """

    ND = 100
    NR = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        randoms = np.random.random((NR, 3))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.linspace(0.001, 0.3, 5)

    pi_max = 0.2

    random_orientation_1 = np.random.random((len(sample1), 2))
    random_orientation_2 = np.random.random((len(sample1), 2))
    random_ellipticities = np.random.random((len(sample1)))

    # analytic randoms
    result_1 = gi_minus_projected(sample1, random_orientation_1, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=1)

    result_2 = gi_minus_projected(sample1, random_orientation_2, random_ellipticities, sample1,
        rp_bins, pi_max, period=period, num_threads=1)
    
    assert not np.allclose(result_1, result_2)