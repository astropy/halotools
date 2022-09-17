"""
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..mean_los_velocity_vs_rp import mean_los_velocity_vs_rp
from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = (
    "test_mean_los_velocity_vs_rp_correctness1",
    "test_mean_los_velocity_vs_rp_correctness2",
    "test_mean_los_velocity_vs_rp_correctness3",
    "test_mean_los_velocity_vs_rp_correctness4",
    "test_mean_los_velocity_vs_rp_parallel",
    "test_mean_los_velocity_vs_rp_auto_consistency",
    "test_mean_los_velocity_vs_rp_cross_consistency",
)

fixed_seed = 43


def pure_python_mean_los_velocity_vs_rp(
    sample1, velocities1, sample2, velocities2, rp_min, rp_max, pi_max, Lbox=None
):
    """Brute force pure python function calculating mean los velocities
    in a single bin of separation.
    """
    if Lbox is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    else:
        xperiod, yperiod, zperiod = Lbox, Lbox, Lbox

    npts1, npts2 = len(sample1), len(sample2)

    running_tally = []
    for i in range(npts1):
        for j in range(npts2):
            dx = sample1[i, 0] - sample2[j, 0]
            dy = sample1[i, 1] - sample2[j, 1]
            dz = sample1[i, 2] - sample2[j, 2]
            dvz = velocities1[i, 2] - velocities2[j, 2]

            if dx > xperiod / 2.0:
                dx = xperiod - dx
            elif dx < -xperiod / 2.0:
                dx = -(xperiod + dx)

            if dy > yperiod / 2.0:
                dy = yperiod - dy
            elif dy < -yperiod / 2.0:
                dy = -(yperiod + dy)

            if dz > zperiod / 2.0:
                dz = zperiod - dz
                zsign_flip = -1
            elif dz < -zperiod / 2.0:
                dz = -(zperiod + dz)
                zsign_flip = -1
            else:
                zsign_flip = 1

            d_rp = np.sqrt(dx * dx + dy * dy)

            if (d_rp > rp_min) & (d_rp < rp_max) & (abs(dz) < pi_max):
                if abs(dz) > 0:
                    vlos = dvz * dz * zsign_flip / abs(dz)
                else:
                    vlos = dvz
                running_tally.append(vlos)

    if len(running_tally) > 0:
        return np.mean(running_tally)
    else:
        return 0.0


def test_mean_radial_velocity_vs_r_vs_brute_force_pure_python():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns
    results that agree with a brute force pure python implementation
    for a random distribution of points, both with and without PBCs.
    """

    npts = 99

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        sample2 = np.random.random((npts, 3))
        velocities1 = np.random.uniform(-10, 10, npts * 3).reshape((npts, 3))
        velocities2 = np.random.uniform(-10, 10, npts * 3).reshape((npts, 3))

    rp_bins, pi_max = np.array([0, 0.1, 0.2, 0.3]), 0.1

    ############################################
    # Run the test with PBCs turned off
    s1s2 = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    rmin, rmax = rp_bins[0], rp_bins[1]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max
    )
    assert np.allclose(s1s2[0], pure_python_s1s2, rtol=0.01)

    rmin, rmax = rp_bins[1], rp_bins[2]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max
    )
    assert np.allclose(s1s2[1], pure_python_s1s2, rtol=0.01)

    rmin, rmax = rp_bins[2], rp_bins[3]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max
    )
    assert np.allclose(s1s2[2], pure_python_s1s2, rtol=0.01)

    # ############################################
    # # Run the test with PBCs operative
    s1s2 = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
        period=1,
    )

    rmin, rmax = rp_bins[0], rp_bins[1]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max, Lbox=1
    )
    assert np.allclose(s1s2[0], pure_python_s1s2, rtol=0.01)

    rmin, rmax = rp_bins[1], rp_bins[2]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max, Lbox=1
    )
    assert np.allclose(s1s2[1], pure_python_s1s2, rtol=0.01)

    rmin, rmax = rp_bins[2], rp_bins[3]
    pure_python_s1s2 = pure_python_mean_los_velocity_vs_rp(
        sample1, velocities1, sample2, velocities2, rmin, rmax, pi_max, Lbox=1
    )
    assert np.allclose(s1s2[2], pure_python_s1s2, rtol=0.01)


def test_mean_los_velocity_vs_rp_correctness1():
    """This function tests that the
    `~halotools.mock_observables.mean_los_velocity_vs_rp` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (1, 0, 0.1), the second at (1, 0.2, 0.25).
    The first set of points is moving at +50 in the z-direction;
    the second set of points is at rest.

    PBCs are set to infinity in this test.

    So in this configuration, the two sets of points are moving towards each other,
    and so the relative z-velocity should be -50 for cross-correlations
    in separation bins containing the pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.

    The tests will be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = -50

    npts = 100

    xc1, yc1, zc1 = 1, 0, 0.1
    xc2, yc2, zc2 = 1, 0.2, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 2] = 50.0

    rp_bins, pi_max = np.array([0, 0.1, 0.15, 0.21, 0.25]), 0.2

    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    assert np.allclose(s1s1[0:2], 0, rtol=0.01)
    assert np.allclose(s1s2[0:2], 0, rtol=0.01)
    assert np.allclose(s2s2[0:2], 0, rtol=0.01)
    assert np.allclose(s1s1[2], 0, rtol=0.01)
    assert np.allclose(s1s2[2], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[2], 0, rtol=0.01)
    assert np.allclose(s1s1[3], 0, rtol=0.01)
    assert np.allclose(s1s2[3], 0, rtol=0.01)
    assert np.allclose(s2s2[3], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max)
    assert np.allclose(s1s1[0:2], 0, rtol=0.01)
    assert np.allclose(s1s1[2], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s1s1[3], 0, rtol=0.01)


def test_mean_los_velocity_vs_rp_correctness2():
    """This function tests that the
    `~halotools.mock_observables.mean_los_velocity_vs_rp` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.35, 0.2).
    The first set of points is moving at -50 in the z-direction;
    the second set of points is at rest.

    PBCs are set to infinity in this test.

    So in this configuration, the two sets of points are moving away from each other,
    and so the relative z-velocity should be +50 for cross-correlations
    in separation bins containing the pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.

    The tests will be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = +50

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.35, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 2] = -50.0

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)


def test_mean_los_velocity_vs_rp_correctness3():
    """This function tests that the
    `~halotools.mock_observables.mean_los_velocity_vs_rp` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.55, 0.1), the second at (0.5, 0.4, 0.95).
    The first set of points is moving at (-50, -10, +20),
    the second set of points is moving at (+25, +10, +40).

    So in this configuration, the second set of points is "gaining ground" on
    the second set in the z-direction, and so the relative z-velocity
    should be -20 for cross-correlations in separation bins containing the pair of points.
    For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.

    The tests will be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = -20

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.55, 0.1
    xc2, yc2, zc2 = 0.5, 0.4, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 0] = -50.0
    velocities1[:, 1] = -10.0
    velocities1[:, 2] = +20.0

    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2[:, 0] = +25.0
    velocities2[:, 1] = +10.0
    velocities2[:, 2] = +40.0

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)


def test_mean_los_velocity_vs_rp_correctness4():
    """This function tests that the
    `~halotools.mock_observables.mean_los_velocity_vs_rp` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.05, 0.05, 0.3), the second at (0.95, 0.95, 0.4).
    The first set of points is moving at (-50, -10, +20),
    the second set of points is moving at (+25, +10, +40).

    So in this configuration, the first set of points is "losing ground" on
    the second set in the z-direction, and so the relative z-velocity
    should be +20 for cross-correlations in separation bins containing the pair of points.
    For any separation bin containing only one set or the other,
    the auto-correlations should be 0 because each set of
    points moves coherently.

    Note that in this test, PBCs operate in both x & y directions
    to identify pairs of points, but PBCs are irrelevant in the z-direction.

    The tests will be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = +20

    npts = 100

    xc1, yc1, zc1 = 0.05, 0.05, 0.3
    xc2, yc2, zc2 = 0.95, 0.95, 0.4

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 0] = -50.0
    velocities1[:, 1] = -10.0
    velocities1[:, 2] = +20.0

    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2[:, 0] = +25.0
    velocities2[:, 1] = +10.0
    velocities2[:, 2] = +40.0

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1, s1s2, s2s2 = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s2s2[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)
    assert np.allclose(s2s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_los_velocity_vs_rp(sample, velocities, rp_bins, pi_max, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)


def test_mean_los_velocity_vs_rp_parallel():
    """
    Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function
    returns identical results for a random distribution of points whether the function
    runs in parallel or serial.
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1_parallel, s1s2_parallel, s2s2_parallel = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=2,
        period=1,
    )

    s1s1_serial, s1s2_serial, s2s2_serial = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=1,
        period=1,
    )

    assert np.allclose(s1s1_serial, s1s1_parallel, rtol=0.001)
    assert np.allclose(s1s2_serial, s1s2_parallel, rtol=0.001)
    assert np.allclose(s2s2_serial, s2s2_parallel, rtol=0.001)


def test_mean_los_velocity_vs_rp_auto_consistency():
    """Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function
    returns self-consistent auto-correlation results
    regardless of whether we ask for cross-correlations.
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s1b, s2s2b = mean_los_velocity_vs_rp(
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


def test_mean_los_velocity_vs_rp_cross_consistency():
    """Verify that the `~halotools.mock_observables.mean_los_velocity_vs_rp` function
    returns self-consistent cross-correlation results
    regardless of whether we ask for auto-correlations.
    """
    npts = 101

    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
        velocities1 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )
        sample2 = np.random.random((npts, 3))
        velocities2 = np.random.normal(loc=0, scale=100, size=npts * 3).reshape(
            (npts, 3)
        )

    rp_bins, pi_max = np.array([0, 0.1, 0.3]), 0.2

    s1s1a, s1s2a, s2s2a = mean_los_velocity_vs_rp(
        sample1, velocities1, rp_bins, pi_max, sample2=sample2, velocities2=velocities2
    )
    s1s2b = mean_los_velocity_vs_rp(
        sample1,
        velocities1,
        rp_bins,
        pi_max,
        sample2=sample2,
        velocities2=velocities2,
        do_auto=False,
    )

    assert np.allclose(s1s2a, s1s2b, rtol=0.001)
