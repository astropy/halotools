"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from .pure_python_mean_radial_velocity_vs_r import pure_python_mean_radial_velocity_vs_r

from ..mean_radial_velocity_vs_r import mean_radial_velocity_vs_r

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ("test_mean_radial_velocity_vs_r1",)

fixed_seed = 43


def test_mean_radial_velocity_vs_r1():
    """Compare <Vr> calculation to simple configuration
    with exactly calculable result
    """
    npts = 10

    xc1, yc1, zc1 = 0.1, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0.0, 0.0, 0.0
    vx2, vy2, vz2 = 20.0, 0.0, 0.0
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    rbins = np.array([0, 0.1, 0.2, 0.3])

    ###########
    # Run the test with PBCs turned off
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(result, [-20, 0, 0])

    # Result should be identical with PBCs turned on
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(result, [-20, 0, 0])


def test_mean_radial_velocity_vs_r2():
    """Compare <Vr> calculation to simple configuration
    with exactly calculable result
    """
    npts = 10

    xc1, yc1, zc1 = 0.05, 0.5, 0.5
    xc2, yc2, zc2 = 0.95, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0.0, 0.0, 0.0
    vx2, vy2, vz2 = 20.0, 0.0, 0.0
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    rbins = np.array([0, 0.05, 0.2, 0.3])

    ###########
    # Run the test with PBCs turned off
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(result, [0, 0, 0])

    # Result should change with PBCs turned on
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(result, [0, -20, 0])


def test_mean_radial_velocity_vs_r3():
    """Brute force comparison of <Vr> calculation to pure python implementation,
    with PBCs turned off, and cross-correlation is tested
    """
    npts1, npts2 = 150, 151
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        velocities1 = np.random.uniform(-100, 100, npts1 * 3).reshape((npts1, 3))
        velocities2 = np.random.uniform(-100, 100, npts2 * 3).reshape((npts2, 3))

    rbins = np.array([0, 0.05, 0.2, 0.3])

    cython_result_no_pbc = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )

    for i, rmin, rmax in zip(range(len(rbins)), rbins[:-1], rbins[1:]):
        python_result_no_pbc = pure_python_mean_radial_velocity_vs_r(
            sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=float("inf")
        )
        assert np.allclose(cython_result_no_pbc[i], python_result_no_pbc)


def test_mean_radial_velocity_vs_r4():
    """Brute force comparison of <Vr> calculation to pure python implementation,
    with PBCs turned on, and cross-correlation is tested
    """
    npts1, npts2 = 150, 151
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        velocities1 = np.random.uniform(-100, 100, npts1 * 3).reshape((npts1, 3))
        velocities2 = np.random.uniform(-100, 100, npts2 * 3).reshape((npts2, 3))

    rbins = np.array([0, 0.05, 0.2, 0.3])

    cython_result_pbc = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1.0,
    )

    for i, rmin, rmax in zip(range(len(rbins)), rbins[:-1], rbins[1:]):
        python_result_no_pbc = pure_python_mean_radial_velocity_vs_r(
            sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=1
        )
        assert np.allclose(cython_result_pbc[i], python_result_no_pbc)


def test_mean_radial_velocity_vs_r5():
    """Brute force comparison of <Vr> calculation to pure python implementation,
    with PBCs turned off, and auto-correlation is tested
    """
    npts1, npts2 = 150, 151
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        velocities1 = np.random.uniform(-100, 100, npts1 * 3).reshape((npts1, 3))
        velocities2 = np.random.uniform(-100, 100, npts2 * 3).reshape((npts2, 3))
    sample1 = np.concatenate((sample1, sample2))
    velocities1 = np.concatenate((velocities1, velocities2))

    rbins = np.array([0, 0.05, 0.2, 0.3])

    cython_result_no_pbc = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins_absolute=rbins
    )

    for i, rmin, rmax in zip(range(len(rbins)), rbins[:-1], rbins[1:]):
        python_result_no_pbc = pure_python_mean_radial_velocity_vs_r(
            sample1, velocities1, sample1, velocities1, rmin, rmax, Lbox=float("inf")
        )
        assert np.allclose(cython_result_no_pbc[i], python_result_no_pbc)


@pytest.mark.installation_test
def test_mean_radial_velocity_vs_r6():
    """Brute force comparison of <Vr> calculation to pure python implementation,
    with PBCs turned on, and auto-correlation is tested
    """
    npts1, npts2 = 150, 151
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        velocities1 = np.random.uniform(-100, 100, npts1 * 3).reshape((npts1, 3))
        velocities2 = np.random.uniform(-100, 100, npts2 * 3).reshape((npts2, 3))
    sample1 = np.concatenate((sample1, sample2))
    velocities1 = np.concatenate((velocities1, velocities2))

    rbins = np.array([0, 0.05, 0.2, 0.3])

    cython_result_no_pbc = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins_absolute=rbins, period=1
    )

    for i, rmin, rmax in zip(range(len(rbins)), rbins[:-1], rbins[1:]):
        python_result_no_pbc = pure_python_mean_radial_velocity_vs_r(
            sample1, velocities1, sample1, velocities1, rmin, rmax, Lbox=1
        )
        assert np.allclose(cython_result_no_pbc[i], python_result_no_pbc)


def test_mean_radial_velocity_vs_r1a():
    """Compare <Vr> calculation to simple configuration
    with exactly calculable result.

    Here we verify that we get identical results when using the
    ``normalize_rbins_by`` feature with unit-normalization.
    """
    npts = 10

    xc1, yc1, zc1 = 0.1, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0.0, 0.0, 0.0
    vx2, vy2, vz2 = 20.0, 0.0, 0.0
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    normalize_rbins_by = np.ones(sample1.shape[0])
    rbins = np.array([0, 0.1, 0.2, 0.3])

    ###########
    # Run the test with PBCs turned off
    result1 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )
    result2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(result1, result2)

    # Result should be identical with PBCs turned on
    result1 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    result2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(result1, result2)


def test_mean_radial_velocity_vs_r1b():
    """Compare <Vr> calculation to simple configuration
    with exactly calculable result, explicitly testing
    a nontrivial example of ``normalize_rbins_by`` feature

    """
    npts = 10

    xc1, yc1, zc1 = 0.2, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0.0, 0.0, 0.0
    vx2, vy2, vz2 = 20.0, 0.0, 0.0
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    normalize_rbins_by = np.zeros(sample1.shape[0]) + 0.1
    rbins_normalized = np.array((0.0, 1.0, 2.0))

    ###########
    # Run the test with PBCs turned off
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins_normalized,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(
        result, [0, -20]
    ), "normalize_rbins_by feature is not implemented correctly"


def test_rvir_normalization_feature():
    """Test the rvir normalization feature. Lay down a regular grid of sample1 points.
    Generate a sample2 point for each point in sample1, at a z value equal to 2.5*Rvir,
    where Rvir is close to 0.01 for every point. Assign the same z-velocity to each sample2 point.
    This allows the value of <Vr> to be calculated simply from <Vz>.
    """
    sample1 = generate_3d_regular_mesh(5)
    velocities1 = np.zeros_like(sample1)
    rvir = 0.01
    sample2 = np.copy(sample1)
    sample2[:, 2] += 2.5 * rvir
    velocities2 = np.zeros_like(sample2)
    velocities2[:, 2] = -43.0

    normalize_rbins_by = np.random.uniform(0.95 * rvir, 1.05 * rvir, sample1.shape[0])

    rbins_normalized = np.array((0.0, 1.0, 2.0, 3.0, 4.0))

    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins_normalized,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
    )
    correct_result = [0, 0, -43, 0]
    assert np.allclose(result, correct_result)


def test_mean_radial_velocity_vs_r2b():
    """Compare <Vr> calculation to simple configuration
    with exactly calculable result
    """
    npts = 10

    xc1, yc1, zc1 = 0.05, 0.5, 0.5
    xc2, yc2, zc2 = 0.95, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0.0, 0.0, 0.0
    vx2, vy2, vz2 = 20.0, 0.0, 0.0
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    normalize_rbins_by = np.zeros(sample1.shape[0]) + 0.1
    rbins_normalized = np.array((0.0, 1.0, 2.0))

    ###########
    # Run the test with PBCs turned off
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins_normalized,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(result, [0, 0])

    # Result should change with PBCs turned on
    result = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_normalized=rbins_normalized,
        normalize_rbins_by=normalize_rbins_by,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(result, [0, -20])


def test_mean_radial_velocity_vs_r_correctness1():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.1), the second at (0.5, 0.5, 0.25).
    The first set of points is moving at +50 in the z-direction;
    the second set of points is at rest.

    PBCs are set to infinity in this test.

    So in this configuration, the two sets of points are moving towards each other,
    and so the radial component of the relative velocity
    should be -50 for cross-correlations in the radial separation bin containing the
    pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.

    The tests will be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = -50

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.1
    xc2, yc2, zc2 = 0.5, 0.5, 0.25

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 2] = 50.0

    rbins = np.array([0, 0.1, 0.3])

    s1s2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins_absolute=rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness2():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.5, 0.05), the second at (0.5, 0.5, 0.95).
    The first set of points is moving at +50 in the z-direction;
    the second set of points is at rest.

    So in this configuration, when PBCs are operative
    the two sets of points are moving away from each other,
    and so the radial component of the relative velocity
    should be +50 for cross-correlations in the radial separation bin containing the
    pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.
    When PBCs are turned off, the function should return zero as the points
    are too distant to find pairs.

    These tests will be applied with and without periodic boundary conditions.
    The tests will also be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = +50

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.5, 0.05
    xc2, yc2, zc2 = 0.5, 0.5, 0.9

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 2] = 50.0

    rbins = np.array([0, 0.1, 0.3])

    # First run the calculation with PBCs set to unity
    s1s2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)

    # Now set PBCs to infinity and verify that we instead get zeros
    s1s2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins_absolute=rbins,
        sample2=sample2,
        velocities2=velocities2,
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)

    # Bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Now repeat the above tests, with and without PBCs
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins_absolute=rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins_absolute=rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness3():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.95, 0.5, 0.5), the second at (0.05, 0.5, 0.5).
    The first set of points is moving at +50 in the x-direction;
    the second set of points is moving at +25 in the x-direction.

    So in this configuration, when PBCs are operative
    the two sets of points are moving towards each other,
    as the first set of points is "gaining ground" on the second set in the x-direction.
    So the radial component of the relative velocity
    should be -25 for cross-correlations in the radial separation bin containing the
    pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.
    When PBCs are turned off, the function should return zero as the points
    are too distant to find pairs.

    These tests will be applied with and without periodic boundary conditions.
    The tests will also be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = -25

    npts = 100

    xc1, yc1, zc1 = 0.95, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 0] = 50.0
    velocities2[:, 0] = 25.0

    rbins = np.array([0, 0.05, 0.3])

    s1s2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        period=1,
        approx_cell1_size=0.1,
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s2 = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        approx_cell2_size=0.1,
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness4():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.5, 0.95, 0.5), the second at (0.5, 0.05, 0.5).
    The first set of points is moving at -50 in the y-direction;
    the second set of points is moving at +25 in the y-direction.

    So in this configuration, when PBCs are operative
    the two sets of points are each moving away from each other,
    so the radial component of the relative velocity
    should be +75 for cross-correlations in the radial separation bin containing the
    pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.
    When PBCs are turned off, the function should return zero as the points
    are too distant to find pairs.

    These tests will be applied with and without periodic boundary conditions.
    The tests will also be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = +75

    npts = 100

    xc1, yc1, zc1 = 0.5, 0.95, 0.5
    xc2, yc2, zc2 = 0.5, 0.05, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, 1] = -50.0
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2[:, 1] = 25.0

    rbins = np.array([0, 0.05, 0.3])

    s1s2 = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins, sample2=sample2, velocities2=velocities2, period=1
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s2 = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins, sample2=sample2, velocities2=velocities2
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_correctness5():
    """This function tests that the
    `~halotools.mock_observables.mean_radial_velocity_vs_r` function returns correct
    results for a controlled distribution of points whose mean radial velocity
    is analytically calculable.

    For this test, the configuration is two tight localizations of points,
    the first at (0.05, 0.05, 0.05), the second at (0.95, 0.95, 0.95).
    The first set of points is moving at (+50, +50, +50);
    the second set of points is moving at (-50, -50, -50).

    So in this configuration, when PBCs are operative
    the two sets of points are each moving towards each other,
    so the radial component of the relative velocity
    should be +100*sqrt(3) for cross-correlations in the radial separation bin containing the
    pair of points. For any separation bin containing only
    one set or the other, the auto-correlations should be 0 because each set of
    points moves coherently.
    When PBCs are turned off, the function should return zero as the points
    are too distant to find pairs.

    These tests will be applied with and without periodic boundary conditions.
    The tests will also be run with the two point configurations passed in as
    separate ``sample1`` and ``sample2`` distributions, as well as bundled
    together into the same distribution.

    """
    correct_relative_velocity = np.sqrt(3) * 100.0

    npts = 91
    xc1, yc1, zc1 = 0.05, 0.05, 0.05
    xc2, yc2, zc2 = 0.95, 0.95, 0.95

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, :] = 50.0
    velocities2[:, :] = -50.0

    rbins = np.array([0, 0.1, 0.3])

    s1s2 = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins, sample2=sample2, velocities2=velocities2, period=1
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], correct_relative_velocity, rtol=0.01)

    # repeat the test but with PBCs set to infinity
    s1s2 = mean_radial_velocity_vs_r(
        sample1, velocities1, rbins, sample2=sample2, velocities2=velocities2
    )
    assert np.allclose(s1s2[0], 0, rtol=0.01)
    assert np.allclose(s1s2[1], 0, rtol=0.01)

    # Now bundle sample2 and sample1 together and only pass in the concatenated sample
    sample = np.concatenate((sample1, sample2))
    velocities = np.concatenate((velocities1, velocities2))

    # Repeat the above tests
    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins, period=1)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], correct_relative_velocity, rtol=0.01)

    s1s1 = mean_radial_velocity_vs_r(sample, velocities, rbins)
    assert np.allclose(s1s1[0], 0, rtol=0.01)
    assert np.allclose(s1s1[1], 0, rtol=0.01)


def test_mean_radial_velocity_vs_r_parallel1():
    """
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function
    returns identical results for two tight loci of points whether the function
    runs in parallel or serial.
    """

    npts = 91
    xc1, yc1, zc1 = 0.5, 0.05, 0.05
    xc2, yc2, zc2 = 0.45, 0.05, 0.1

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts * 3).reshape(npts, 3)
    velocities2 = np.zeros(npts * 3).reshape(npts, 3)
    velocities1[:, :] = 50.0
    velocities2[:, :] = 0.0

    rbins = np.array([0, 0.1, 0.3])

    s1s2_parallel = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=2,
        period=1,
    )

    s1s2_serial = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=1,
        period=1,
    )

    assert np.all(s1s2_serial == s1s2_parallel)


def test_mean_radial_velocity_vs_r_parallel2():
    """
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function
    returns identical results for two random distributions of points whether the function
    runs in parallel or serial, with PBCs operative.
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

    rbins = np.array([0, 0.1, 0.3])

    s1s2_parallel = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=2,
        period=1,
    )

    s1s2_serial = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=1,
        period=1,
    )

    assert np.allclose(s1s2_serial, s1s2_parallel, rtol=0.001)


def test_mean_radial_velocity_vs_r_parallel3():
    """
    Verify that the `~halotools.mock_observables.mean_radial_velocity_vs_r` function
    returns identical results for two random distributions of points whether the function
    runs in parallel or serial, with PBCs turned off.
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

    rbins = np.array([0, 0.1, 0.3])

    s1s2_parallel = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=2,
    )

    s1s2_serial = mean_radial_velocity_vs_r(
        sample1,
        velocities1,
        rbins,
        sample2=sample2,
        velocities2=velocities2,
        num_threads=1,
    )

    assert np.allclose(s1s2_serial, s1s2_parallel, rtol=0.001)
