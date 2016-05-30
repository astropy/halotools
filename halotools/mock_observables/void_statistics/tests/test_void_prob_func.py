""" Module providing unit-testing for the
`~halotools.mock_observables.void_prob_func` function.
"""
from __future__ import (absolute_import, division, print_function)
import numpy as np
from astropy.tests.helper import pytest

from ..void_prob_func import void_prob_func

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_vpf1', 'test_vpf2', 'test_vpf3')


def test_vpf1():
    """ Verify that the VPF raises no exceptions
    for several reasonable choices of rbins.

    period = [1, 1, 1]
    """

    Npts = 100
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    sample1 = np.random.random((Npts, 3))
    n_ran = 100

    rbins = np.logspace(-2, -1, 20)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period)

    rbins = np.linspace(0.1, 0.3, 10)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period)


def test_vpf2():
    """ Verify that the VPF raises no exceptions
    for several reasonable choices of rbins.

    period = None
    """

    Npts = 100
    Lbox = 1
    period = None
    sample1 = np.random.random((Npts, 3))
    n_ran = 100

    rbins = np.logspace(-2, -1, 20)
    vpf = void_prob_func(sample1, rbins, n_ran, period)


@pytest.mark.slow
def test_vpf3():
    """ Verify that the VPF returns consistent results
    regardless of the value of approx_cell1_size.
    """
    np.random.seed(43)

    Npts = 1000
    Lbox = 1
    period = Lbox
    sample1 = np.random.random((Npts, 3))
    n_ran = 1000

    rbins = np.logspace(-2, -1, 5)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period)
    vpf2 = void_prob_func(sample1, rbins, n_ran=n_ran, period=period,
        approx_cell1_size=[0.2, 0.2, 0.2])
    assert np.allclose(vpf, vpf2, rtol=0.1)
