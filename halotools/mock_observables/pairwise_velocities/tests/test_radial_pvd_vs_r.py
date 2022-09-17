"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..radial_pvd_vs_r import radial_pvd_vs_r

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ("test_radial_pvd_vs_r_correctness1",)

fixed_seed = 43


def test_radial_pvd_vs_r_correctness1():
    """This function tests that the
    `~halotools.mock_observables.radial_pvd_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.25).
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
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rbins = np.array([0.001, 0.1, 0.3])

    s1s2 = radial_pvd_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)


def test_radial_pvd_vs_r_correctness2():
    """This function tests that the
    `~halotools.mock_observables.radial_pvd_vs_r` function returns correct
    results for a controlled distribution of points whose radial velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.95).
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
    xc2, yc2, zc2 = 0.5, 0.5, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rbins = np.array([0.001, 0.1, 0.3])
    s1s2 = radial_pvd_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s2[0], 0, rtol=0.1)
    assert np.allclose(s1s2[1], correct_cross_pvd, rtol=0.001)


def test_radial_pvd_vs_r_correctness3():
    """This function tests that the
    `~halotools.mock_observables.radial_pvd_vs_r` function returns correct
    results for a controlled distribution of points whose radial velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.25).
    The first set of points is moving with random uniform z-velocities;
    the second set of points is at rest.

    PBCs are turned off in this test.

    For every velocity in sample1, since we can count pairs analytically
    for this configuration we know exactly how many appearances of each
    velocity there will be, so we can calculate np.std on the exact
    same set of points as the marked pair-counter should operate on.

    This is the same test as test_radial_pvd_vs_r_correctness3, only here we
    bundle the two sets of points into the same sample.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1 = radial_pvd_vs_r(sample, velocities, rbins_absolute=rbins)

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s1[1], correct_cross_pvd, rtol=0.001)


def test_radial_pvd_vs_r_correctness4():
    """This function tests that the
    `~halotools.mock_observables.radial_pvd_vs_r` function returns correct
    results for a controlled distribution of points whose radial velocity
    can be simply calculated.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.95).
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
    xc2, yc2, zc2 = 0.5, 0.5, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    rbins = np.array([0.001, 0.1, 0.3])
    s1s1 = radial_pvd_vs_r(sample, velocities, rbins_absolute=rbins, period=1)

    correct_cross_pvd = np.std(np.repeat(velocities1[:, 2], npts))

    assert np.allclose(s1s1[1], correct_cross_pvd, rtol=0.001)


def test_radial_pvd_vs_r1():
    """Verify that different choices for the cell size has no
    impact on the results.
    """

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rbins = np.linspace(0, 0.3, 10)
    result1 = radial_pvd_vs_r(sample1, velocities1, rbins_absolute=rbins)
    result2 = radial_pvd_vs_r(
        sample1, velocities1, rbins_absolute=rbins, approx_cell1_size=[0.2, 0.2, 0.2]
    )
    assert np.allclose(result1, result2, rtol=0.0001)


def test_radial_pvd_vs_r_auto_consistency():
    """Verify that we get self-consistent auto-correlation results
    regardless of how we do the cross-correlation.
    """
    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    with NumpyRNGContext(fixed_seed):
        velocities1[:, 2] = np.random.uniform(0, 1, npts)

    rbins = np.linspace(0, 0.3, 10)
    s1s1a = radial_pvd_vs_r(sample1, velocities1, rbins_absolute=rbins)
    s1s1b = radial_pvd_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample1,
        velocities2=velocities1,
    )

    assert np.allclose(s1s1a, s1s1b, rtol=0.001)
