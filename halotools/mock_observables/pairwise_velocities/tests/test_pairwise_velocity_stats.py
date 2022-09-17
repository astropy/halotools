"""
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..los_pvd_vs_rp import los_pvd_vs_rp
from ..mean_los_velocity_vs_rp import mean_los_velocity_vs_rp

__all__ = ("test_mean_los_velocity_vs_rp_auto_consistency",)

fixed_seed = 43


def test_mean_los_velocity_vs_rp_auto_consistency():

    npts = 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.rand(npts, 3)
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.rand(npts, 3)
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rbins = np.linspace(0, 0.3, 10)
    pi_max = 0.2
    s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(
        sample1, velocities1, rbins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s1b, s2s2b = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rbins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_cross=False,
    )

    assert np.allclose(s1s1a, s1s1b, rtol=0.001)
    assert np.allclose(s2s2a, s2s2b, rtol=0.001)


def test_mean_los_velocity_vs_rp_cross_consistency():

    npts = 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.rand(npts, 3)
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.rand(npts, 3)
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rbins = np.linspace(0, 0.3, 10)
    pi_max = 0.3
    s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(
        sample1, velocities1, rbins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s2b = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rbins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    assert np.allclose(s1s2a, s1s2b, rtol=0.001)


def test_los_pvd_vs_rp_auto_consistency():

    npts = 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.rand(npts, 3)
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.rand(npts, 3)
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rbins = np.linspace(0, 0.3, 10)
    pi_max = 0.2
    s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(
        sample1, velocities1, rbins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s1b, s2s2b = los_pvd_vs_rp(
        sample1,
        velocities1,
        rbins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_cross=False,
    )

    assert np.allclose(s1s1a, s1s1b, rtol=0.001)
    assert np.allclose(s2s2a, s2s2b, rtol=0.001)


def test_los_pvd_vs_rp_cross_consistency():

    npts = 200
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.rand(npts, 3)
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.rand(npts, 3)
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rbins = np.linspace(0, 0.3, 10)
    pi_max = 0.3
    s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(
        sample1, velocities1, rbins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s2b = los_pvd_vs_rp(
        sample1,
        velocities1,
        rbins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    assert np.allclose(s1s2a, s1s2b, rtol=0.001)
