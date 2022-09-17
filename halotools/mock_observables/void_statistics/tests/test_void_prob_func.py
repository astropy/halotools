""" Module providing unit-testing for the
`~halotools.mock_observables.void_prob_func` function.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..void_prob_func import void_prob_func

from ...tests.cf_helpers import generate_locus_of_3d_points

from ....custom_exceptions import HalotoolsError

__all__ = ("test_vpf1", "test_vpf2", "test_vpf3")

fixed_seed = 43


def test_vpf1():
    """Verify that the VPF raises no exceptions
    for several reasonable choices of rbins.

    period = [1, 1, 1]
    """

    Npts = 100
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
    n_ran = 100

    rbins = np.logspace(-2, -1, 20)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period, seed=fixed_seed)

    rbins = np.linspace(0.1, 0.3, 10)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period, seed=fixed_seed)


def test_vpf2():
    """Verify that the VPF raises no exceptions
    for several reasonable choices of rbins.

    period = None
    """

    Npts = 100
    Lbox = 1
    period = None
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
    n_ran = 100

    rbins = np.logspace(-2, -1, 20)
    vpf = void_prob_func(sample1, rbins, n_ran, period, seed=fixed_seed)


def test_vpf3():
    """Verify that the VPF returns consistent results
    regardless of the value of approx_cell1_size.
    """
    Npts = 1000
    Lbox = 1
    period = Lbox
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
    n_ran = 1000

    rbins = np.logspace(-2, -1, 5)
    vpf = void_prob_func(sample1, rbins, n_ran=n_ran, period=period, seed=fixed_seed)
    vpf2 = void_prob_func(
        sample1,
        rbins,
        n_ran=n_ran,
        period=period,
        approx_cell1_size=[0.2, 0.2, 0.2],
        seed=fixed_seed,
    )
    assert np.allclose(vpf, vpf2, rtol=0.01)


def test_vpf_process_args1():
    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        random_sphere_centers = np.random.random((Npts, 3))
    n_ran = 1000
    rbins = np.logspace(-1.5, -1, 5)

    with pytest.raises(HalotoolsError) as err:
        __ = void_prob_func(sample1, rbins, n_ran=n_ran, period=[Lbox, Lbox])
    substr = "Input ``period`` must either be a float or length-3 sequence"
    assert substr in err.value.args[0]


def test_vpf_process_args2():
    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        random_sphere_centers = np.random.random((Npts, 2))
    n_ran = 1000
    rbins = np.logspace(-1.5, -1, 5)

    with pytest.raises(HalotoolsError) as err:
        __ = void_prob_func(
            sample1,
            rbins,
            n_ran=n_ran,
            period=period,
            random_sphere_centers=random_sphere_centers,
        )
    substr = "If passing in ``random_sphere_centers``, do not also pass in ``n_ran``."
    assert substr in err.value.args[0]


def test_vpf_process_args3():
    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        random_sphere_centers = np.random.random((Npts, 2))
    n_ran = 1000
    rbins = np.logspace(-1.5, -1, 5)

    with pytest.raises(HalotoolsError) as err:
        __ = void_prob_func(
            sample1, rbins, period=period, random_sphere_centers=random_sphere_centers
        )
    substr = "Your input ``random_sphere_centers`` must have shape (Nspheres, 3)"
    assert substr in err.value.args[0]


def test_vpf_process_args4():
    Npts = 1000
    Lbox = 1
    period = np.array([Lbox, Lbox, Lbox])
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        random_sphere_centers = np.random.random((Npts, 2))
    n_ran = 1000
    rbins = np.logspace(-1.5, -1, 5)

    with pytest.raises(HalotoolsError) as err:
        __ = void_prob_func(sample1, rbins, period=period)
    substr = "You must pass either ``n_ran`` or ``random_sphere_centers``"
    assert substr in err.value.args[0]
