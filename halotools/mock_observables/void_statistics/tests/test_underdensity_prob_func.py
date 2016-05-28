""" Module providing unit-testing for the
`~halotools.mock_observables.underdensity_prob_func function.
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
from astropy.tests.helper import pytest

from ..underdensity_prob_func import underdensity_prob_func
from ..void_prob_func import void_prob_func

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_upf1', 'test_upf2', 'test_upf3', 'test_upf4')


def test_upf1():
    """ Verify that the UPF raises no exceptions
    for several reasonable choices of rbins.
    """

    Npts = 100
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    sample1 = np.random.random((Npts, 3))
    n_ran = 100

    rbins = np.logspace(-2, -1, 20)
    upf = underdensity_prob_func(sample1, rbins,
        n_ran=n_ran, period=period)

    rbins = np.linspace(0.1, 0.3, 10)
    upf = underdensity_prob_func(sample1, rbins,
        n_ran=n_ran, period=period)


@pytest.mark.slow
def test_upf2():
    """ Verify that the UPF behaves properly when changing the
    density threshold criterion.
    """

    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    sample1 = np.random.random((Npts, 3))
    random_sphere_centers = np.random.random((Npts, 3))

    rbins = np.logspace(-1.5, -1, 5)
    upf = underdensity_prob_func(sample1, rbins,
        random_sphere_centers=random_sphere_centers, period=period, u=0.5)
    upf2 = underdensity_prob_func(sample1, rbins,
        random_sphere_centers=random_sphere_centers, period=period, u=0.00001)
    print(upf)
    print(upf2)
    assert np.all(upf >= upf2)


@pytest.mark.slow
def test_upf3():
    """ Verify that the UPF converges to the VPF in the
    limit of vanishing density threshold.
    """

    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    sample1 = np.random.random((Npts, 3))
    random_sphere_centers = np.random.random((Npts, 3))

    rbins = np.logspace(-2, -0.5, 5)

    upf = underdensity_prob_func(sample1, rbins,
        random_sphere_centers=random_sphere_centers, period=period, u=0)
    upf2 = underdensity_prob_func(sample1, rbins,
        random_sphere_centers=random_sphere_centers, period=period, u=0.001)
    vpf = void_prob_func(sample1, rbins,
        random_sphere_centers=random_sphere_centers, period=period)
    assert np.all(upf == vpf)
    assert np.all(upf2 >= vpf)


@pytest.mark.slow
def test_upf4():
    """ Verify that the UPF and VPF raise no exceptions
    when operating on a tight locus of points.
    """

    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    sample1 = generate_locus_of_3d_points(Npts)
    n_ran = 1000

    rbins = np.logspace(-1.5, -1, 5)
    upf = underdensity_prob_func(sample1, rbins, n_ran=n_ran, period=period)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period)
