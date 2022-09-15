r"""
test module for pair counts in s and mu bins
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

# load pair counters
from ..npairs_s_mu import npairs_s_mu
from ....mock_observables import npairs_3d

# load comparison simple pair counters
from ..pairs import s_mu_npairs as pure_python_brute_force_npairs_s_mu

import pytest
from astropy.utils.misc import NumpyRNGContext

slow = pytest.mark.slow
fixed_seed = 43

__all__ = (
    "test_npairs_s_mu_periodic",
    "test_npairs_s_mu_nonperiodic",
    "test_npairs_s_mu_point_surrounded_by_circle",
)

# set up random points to test pair counters
Npts = 1000
with NumpyRNGContext(fixed_seed):
    random_sample = np.random.random((Npts, 3))
period = np.array([1.0, 1.0, 1.0])
num_threads = 2

# set up a regular grid of points to test pair counters
Npts2 = 10
epsilon = 0.001
gridx = np.linspace(0, 1 - epsilon, Npts2)
gridy = np.linspace(0, 1 - epsilon, Npts2)
gridz = np.linspace(0, 1 - epsilon, Npts2)

xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
grid_points = np.vstack([xx, yy, zz]).T

grid_jackknife_spacing = 0.5
grid_jackknife_ncells = int(1 / grid_jackknife_spacing)
ix = np.floor(gridx / grid_jackknife_spacing).astype(int)
iy = np.floor(gridy / grid_jackknife_spacing).astype(int)
iz = np.floor(gridz / grid_jackknife_spacing).astype(int)
ixx, iyy, izz = np.array(np.meshgrid(ix, iy, iz))
ixx = ixx.flatten()
iyy = iyy.flatten()
izz = izz.flatten()
grid_indices = np.ravel_multi_index(
    [ixx, iyy, izz],
    [grid_jackknife_ncells, grid_jackknife_ncells, grid_jackknife_ncells],
)
grid_indices += 1


def test_npairs_s_mu_periodic():
    r"""
    test npairs_s_mu with periodic boundary conditions on a random set of points.
    """

    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    # count pairs using optimized double tree pair counter
    result = npairs_s_mu(
        random_sample,
        random_sample,
        s_bins,
        mu_bins,
        period=period,
        num_threads=num_threads,
    )

    msg = "The returned result is an unexpected shape."
    assert np.shape(result) == (len(s_bins), N_mu_bins), msg

    # count pairs using 3D pair counter
    test_result = npairs_3d(
        random_sample, random_sample, s_bins, period=period, num_threads=num_threads
    )

    # summing pairs counts along mu axis should match the 3D radial pair result
    result = np.diff(result, axis=1)
    result = np.sum(result, axis=1) + Npts

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg

    # compare to brute force s and mu pair counter on a set of random points
    result = npairs_s_mu(
        random_sample,
        random_sample,
        s_bins,
        mu_bins,
        period=period,
        num_threads=num_threads,
    )

    test_result = pure_python_brute_force_npairs_s_mu(
        random_sample, random_sample, s_bins, mu_bins, period=period
    )

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result == result), msg


def test_npairs_s_mu_nonperiodic():
    r"""
    test npairs_s_mu without periodic boundary conditions on a random set of points.
    """

    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    # count pairs using optimized double tree pair counter
    result = npairs_s_mu(
        random_sample, random_sample, s_bins, mu_bins, num_threads=num_threads
    )

    msg = "The returned result is an unexpected shape."
    assert np.shape(result) == (len(s_bins), N_mu_bins), msg

    # count pairs using 3D pair counter
    test_result = npairs_3d(
        random_sample, random_sample, s_bins, num_threads=num_threads
    )

    # summing pairs counts along mu axis should match the 3D radial pair result
    result = np.diff(result, axis=1)
    result = np.sum(result, axis=1) + Npts

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg

    # compare to brute force s and mu pair counter on a set of random points
    result = npairs_s_mu(
        random_sample, random_sample, s_bins, mu_bins, num_threads=num_threads
    )

    test_result = pure_python_brute_force_npairs_s_mu(
        random_sample, random_sample, s_bins, mu_bins
    )

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(test_result == result), msg


def test_npairs_s_mu_point_surrounded_by_circle():
    r"""
    test npairs_s_mu without periodic boundary conditions on a point surrounded by an
    evenly distributed circle of points aligned with the LOS.
    """

    # put one point per degree in the circle
    theta = np.linspace(0, 2 * np.pi - 2 * np.pi / 360, 360) + (2.0 * np.pi) / (360 * 2)
    r = 1.5
    x = r * np.cos(theta)
    y = r * np.zeros(360)
    z = r * np.sin(theta)
    sample2 = np.vstack((x, y, z)).T

    # find pairs between the circle and the origin
    sample1 = np.array([[0, 0, 0]])

    # define bins
    theta_bins = np.linspace(0.0, np.pi / 2.0, 91)  # make 1 degree mu bins
    mu_bins = np.sort(np.cos(theta_bins))
    s_bins = np.array([0, 1, 2])

    # count pairs using optimized double tree pair counter
    result = npairs_s_mu(sample1, sample2, s_bins, mu_bins, num_threads=num_threads)
    pairs = np.diff(np.diff(result, axis=0), axis=1)

    # note that there should be 4 pairs per mu bin
    # since each quadrant of the circle counts once for each mu bin
    # and there is one point per degree

    msg = "The number of pairs between the origin and a circle with one point per degree is incorrect."
    assert np.all(pairs[1, :] == 4), msg
