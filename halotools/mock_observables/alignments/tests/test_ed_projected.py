"""
Module providing unit-testing for the `~halotools.mock_observables.alignments.w_gplus` function.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..ed_projected import ed_projected

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_shape', 'test_threading', 'test_pbcs', 'test_random_result')

fixed_seed = 43


def test_shape():
    """
    make sure the result that is returned has the correct shape
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        random_orientation = np.random.random((len(sample1), 2))*2 - 1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    pi_max = 0.2

    # analytic randoms
    result_1 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=period, num_threads=1)

    assert np.shape(result_1) == (len(rbins)-1, 1)

    result_2 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=period, num_threads=3)

    assert np.shape(result_2) == (len(rbins)-1, 1)


def test_threading():
    """
    test to make sure the results are consistent when num_threads=1 or >1
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        random_orientation = np.random.random((len(sample1), 2))*2 - 1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    pi_max = 0.2

    result_1 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=period, num_threads=1)

    result_2 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=period, num_threads=3)

    assert np.allclose(result_1, result_2)


def test_pbcs():
    """
    test to make sure the results are consistent with and without PBCs
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        random_orientation = np.random.random((len(sample1), 2))*2 - 1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    pi_max = 0.2

    result_1 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=period, num_threads=1)

    result_2 = ed_projected(sample1, random_orientation, sample1,
        rbins, pi_max, period=None, num_threads=3)
    
    tol = 10.0/ND

    assert np.allclose(result_1, result_2, atol=tol)

@slow
def test_random_result():
    """
    test to make sure the correlation function returns the expected result for a random distribution of orientations
    """

    ND = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((ND, 3))
        sample2 = np.random.random((ND, 3))
        random_orientation1 = np.random.random((len(sample1), 2))*2 - 1.0

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    pi_max = 0.2

    result_1 = ed_projected(sample1, random_orientation1, sample2,
        rbins, pi_max, period=period, num_threads=1)
    
    tol = 10.0/ND

    assert np.allclose(result_1, 0.0, atol=tol)

