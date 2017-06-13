"""
"""
import numpy as np

from .pure_python_mean_radial_velocity_vs_r import pure_python_mean_radial_velocity_vs_r

from ...tests.cf_helpers import generate_locus_of_3d_points

__all__ = ('test_pure_python1', )

fixed_seed = 43


def test_pure_python1():
    """ Verify that the brute-force pairwise velocity function returns the
    correct result for an analytically calculable case.
    """
    correct_relative_velocity = -25

    npts = 100

    xc1, yc1, zc1 = 0.95, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5

    sample1 = generate_locus_of_3d_points(npts, xc=xc1, yc=yc1, zc=zc1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(npts, xc=xc2, yc=yc2, zc=zc2, seed=fixed_seed)

    velocities1 = np.zeros(npts*3).reshape(npts, 3)
    velocities2 = np.zeros(npts*3).reshape(npts, 3)
    velocities1[:, 0] = 50.
    velocities2[:, 0] = 25.

    rbins = np.array([0, 0.05, 0.3])

    msg = "pure python result is incorrect"

    rmin, rmax = rbins[0], rbins[1]
    pure_python_s1s2 = pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=1)
    assert pure_python_s1s2 == 0, msg

    rmin, rmax = rbins[1], rbins[2]
    pure_python_s1s2 = pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=1)
    assert np.allclose(pure_python_s1s2, correct_relative_velocity, rtol=0.01), msg


def test_pure_python2():
    """
    """
    npts = 10
    rmin, rmax = 0., 0.2
    xc1, yc1, zc1 = 0.1, 0.5, 0.5
    xc2, yc2, zc2 = 0.05, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0., 0., 0.
    vx2, vy2, vz2 = 20., 0., 0.
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    result = pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=float('inf'))
    assert np.all(result == -20)


def test_pure_python3():
    """
    """
    npts = 10
    rmin, rmax = 0., 0.2
    xc1, yc1, zc1 = 0.05, 0.5, 0.5
    xc2, yc2, zc2 = 0.1, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0., 0., 0.
    vx2, vy2, vz2 = 20., 0., 0.
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    result = pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=float('inf'))
    assert np.all(result == 20)


def test_pure_python4():
    """
    """
    npts = 10
    rmin, rmax = 0., 0.2
    xc1, yc1, zc1 = 0.05, 0.5, 0.5
    xc2, yc2, zc2 = 0.95, 0.5, 0.5
    sample1 = np.zeros((npts, 3)) + (xc1, yc1, zc1)
    sample2 = np.zeros((npts, 3)) + (xc2, yc2, zc2)

    vx1, vy1, vz1 = 0., 0., 0.
    vx2, vy2, vz2 = 20., 0., 0.
    velocities1 = np.zeros((npts, 3)) + (vx1, vy1, vz1)
    velocities2 = np.zeros((npts, 3)) + (vx2, vy2, vz2)

    result = pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=1)
    assert np.all(result == -20)
