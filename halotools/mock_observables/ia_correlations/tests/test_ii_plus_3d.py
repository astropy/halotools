"""
Module providing unit-testing for the `~halotools.mock_observables.alignments.w_gplus` function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..ii_plus_3d import ii_plus_3d

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_shape', 'test_threading', 'test_round_result', 'test_random_result')

fixed_seed = 43


def test_shape():
    """
    make sure the result that is returned has the correct shape
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        random_orientation1 = np.random.random((len(sample1), 3))*2 - 1.0
        random_ellipticities1 = np.random.random((len(sample1)))

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)

    # analytic randoms
    result_1 = ii_plus_3d(sample1, random_orientation1, random_ellipticities1, sample1,
         random_orientation1, random_ellipticities1, rbins, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rbins)-1, )

    result_2 = ii_plus_3d(sample1, random_orientation1, random_ellipticities1, sample1,
         random_orientation1, random_ellipticities1, rbins, period=period, num_threads=1)

    assert np.shape(result_2) == (len(rbins)-1, )


def test_threading():
    """
    test to make sure the results are consistent when num_threads=1 or >1
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        random_orientation = np.random.random((len(sample1), 3))*2 - 1.0
        random_ellipticities = np.random.random((len(sample1)))

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)

    result_1 = ii_plus_3d(sample1, random_orientation, random_ellipticities, 
        sample1, random_orientation, random_ellipticities, rbins, period=period, num_threads=1)

    result_2 = ii_plus_3d(sample1, random_orientation, random_ellipticities, 
        sample1, random_orientation, random_ellipticities, rbins, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)

def test_round_result():
    """
    Test to make sure the correlation comes out as expected in the case of non-elliptical input.
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        sample2 = np.random.random((ND, 3))
        random_orientation_1 = np.random.random((len(sample1), 3))*2 - 1.0
        random_orientation_2 = np.random.random((len(sample2), 3))*2 - 1.0
        zero_ellipticities = np.zeros((len(sample1)))

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)

    # analytic randoms
    result_1 = ii_plus_3d(sample1, random_orientation_1, zero_ellipticities,
        sample2,  random_orientation_2, zero_ellipticities, rbins, period=period, num_threads=1)

    tol = 2*10.0/ND
    
    assert np.allclose(result_1, 0.0, atol=tol)


@slow
def test_random_result():
    """
    test to make sure the correlation function returns the expected result for a random distribution of orientations
    Note that the expectation of xi_++ are non-zero in the case of random orientations.
    It actually comes out as (1/6)*(1/6)
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        sample2 = np.random.random((ND, 3))
        random_orientation1 = np.random.random((len(sample1), 3))*2 - 1.0
        random_orientation2 = np.random.random((len(sample2), 3))*2 - 1.0
        random_ellipticities1 = np.random.random((len(sample1)))
        random_ellipticities2 = np.random.random((len(sample2)))

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)

    result_1 = ii_plus_3d(sample1, random_orientation1, random_ellipticities1,
        sample2,  random_orientation2, random_ellipticities2, rbins, period=period, num_threads=1)
    
    tol = 2*10.0/ND

    assert np.allclose(result_1, 1./36., atol=tol)

