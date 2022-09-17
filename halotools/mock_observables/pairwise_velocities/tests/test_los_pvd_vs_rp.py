"""
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..los_pvd_vs_rp import los_pvd_vs_rp

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = (
    "test_los_pvd_vs_rp_correctness1",
    "test_los_pvd_vs_rp_correctness2",
    "test_los_pvd_vs_rp_correctness3",
    "test_los_pvd_vs_rp_auto_consistency",
    "test_los_pvd_vs_rp_cross_consistency",
)

fixed_seed = 43


def test_los_pvd_vs_rp_correctness1():
    """This function tests that the
    `~halotools.mock_observables.los_pvd_vs_rp` function returns correct
    results for a controlled distribution of points whose line-of-sight velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.35, 0.25).
    The first set of points is moving with random uniform z-velocities;
    the second set of points is at rest.

    PBCs are set to infinity in this test.

    For every velocity in sample1, since we can count pairs analytically
    for this configuration we know exactly how many appearances of each
    velocity there will be, so we can calculate np.std on the exact
    same set of points as the marked pair-counter should operate on.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rp_bins, pi_max = np.array([0.001, 0.1, 0.3]), 0.2

    s1s2 = los_pvd_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)


def test_los_pvd_vs_rp_correctness2():
    """This function tests that the
    `~halotools.mock_observables.los_pvd_vs_rp` function returns correct
    results for a controlled distribution of points whose line-of-sight velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.35, 0.95).
    The first set of points is moving with random uniform z-velocities;
    the second set of points is at rest.

    PBCs are operative in this test.

    For every velocity in sample1, since we can count pairs analytically
    for this configuration we know exactly how many appearances of each
    velocity there will be, so we can calculate np.std on the exact
    same set of points as the marked pair-counter should operate on.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rp_bins, pi_max = np.array([0.001, 0.1, 0.3]), 0.2

    s1s2 = los_pvd_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
        period=1,
    )

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)


def test_los_pvd_vs_rp_correctness3():
    """This function tests that the
    `~halotools.mock_observables.los_pvd_vs_rp` function returns correct
    results for a controlled distribution of points whose line-of-sight velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.35, 0.95).
    The first set of points is moving with random uniform z-velocities;
    the second set of points is at rest.

    PBCs are operative in this test.

    For every velocity in sample1, since we can count pairs analytically
    for this configuration we know exactly how many appearances of each
    velocity there will be, so we can calculate np.std on the exact
    same set of points as the marked pair-counter should operate on.

    This is the same test as test_los_pvd_vs_rp_correctness2, only here we
    bundle the two sets of points into the same sample.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rp_bins, pi_max = np.array([0.001, 0.1, 0.3]), 0.2

    s1s1 = los_pvd_vs_rp(sample, velocities, rp_bins, pi_max, period=1)

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s1[1], correct_cross_pvd, rtol=0.001)


def test_los_pvd_vs_rp_auto_consistency():
    """Verify that we get self-consistent auto-correlation results
    regardless of whether we ask for cross-correlations.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rp_bins, pi_max = np.array([0.001, 0.1, 0.3]), 0.2

    s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s1b, s2s2b = los_pvd_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_cross=False,
    )

    assert np.allclose(s1s1a, s1s1b, rtol=0.001)
    assert np.allclose(s2s2a, s2s2b, rtol=0.001)


def test_los_pvd_vs_rp_cross_consistency():
    """Verify that we get self-consistent auto-correlation results
    regardless of whether we ask for cross-correlations.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rp_bins, pi_max = np.array([0.001, 0.1, 0.3]), 0.2

    s1s1a, s1s2a, s2s2a = los_pvd_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s2b = los_pvd_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    assert np.allclose(s1s2a, s1s2b, rtol=0.001)
