"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

# load pair counters
from ..npairs_jackknife_xy_z import npairs_jackknife_xy_z
# load comparison simple pair counters

import pytest
from astropy.utils.misc import NumpyRNGContext

slow = pytest.mark.slow

__all__ = ('test_npairs_jackknife_xy_z_periodic', 'test_npairs_jackknife_xy_z_nonperiodic')

fixed_seed = 43

# set up random points to test pair counters
Npts = 1000
with NumpyRNGContext(fixed_seed):
    random_sample = np.random.random((Npts, 3))
period = np.array([1.0, 1.0, 1.0])
num_threads = 2

# set up a regular grid of points to test pair counters
Npts2 = 10
epsilon = 0.001
gridx = np.linspace(0, 1-epsilon, Npts2)
gridy = np.linspace(0, 1-epsilon, Npts2)
gridz = np.linspace(0, 1-epsilon, Npts2)

xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
grid_points = np.vstack([xx, yy, zz]).T

grid_jackknife_spacing = 0.5
grid_jackknife_ncells = int(1/grid_jackknife_spacing)
ix = np.floor(gridx/grid_jackknife_spacing).astype(int)
iy = np.floor(gridy/grid_jackknife_spacing).astype(int)
iz = np.floor(gridz/grid_jackknife_spacing).astype(int)
ixx, iyy, izz = np.array(np.meshgrid(ix, iy, iz))
ixx = ixx.flatten()
iyy = iyy.flatten()
izz = izz.flatten()
grid_indices = np.ravel_multi_index([ixx, iyy, izz],
    [grid_jackknife_ncells, grid_jackknife_ncells, grid_jackknife_ncells])
grid_indices += 1


def test_npairs_jackknife_3d_periodic():
    """
    test npairs_jackknife_3d with periodic boundary conditions.
    """

    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0.0, 0.1, 0.2, 0.3])

    # define the jackknife sample labels
    Npts = len(random_sample)
    N_jsamples = 10
    with NumpyRNGContext(fixed_seed):
        jtags1 = np.sort(np.random.randint(1, N_jsamples+1, size=Npts))

    # define weights
    weights1 = np.random.random(Npts)

    result = npairs_jackknife_xy_z(random_sample, random_sample, rp_bins, pi_bins, period=period,
        jtags1=jtags1, jtags2=jtags1, N_samples=10,
        weights1=weights1, weights2=weights1, num_threads=num_threads)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (N_jsamples+1, len(rp_bins), len(pi_bins)), msg

    # Now verify that when computing jackknife pairs on a regularly spaced grid,
    # the counts in all subvolumes are identical

    grid_result = npairs_jackknife_xy_z(grid_points, grid_points, rp_bins, pi_bins, period=period,
        jtags1=grid_indices, jtags2=grid_indices, N_samples=grid_jackknife_ncells**3,
        num_threads=num_threads)

    for icell in range(1, grid_jackknife_ncells**3-1):
        assert np.all(grid_result[icell, :, :] == grid_result[icell+1, :, :])


def test_npairs_jackknife_xy_z_nonperiodic():
    """
    test npairs_jackknife_3d without periodic boundary conditions.
    """

    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0.0, 0.1, 0.2, 0.3])

    # define the jackknife sample labels
    Npts = len(random_sample)
    N_jsamples = 10
    with NumpyRNGContext(fixed_seed):
        jtags1 = np.sort(np.random.randint(1, N_jsamples+1, size=Npts))
        # define weights
        weights1 = np.random.random(Npts)

    result = npairs_jackknife_xy_z(random_sample, random_sample, rp_bins, pi_bins, period=None,
        jtags1=jtags1, jtags2=jtags1, N_samples=10,
        weights1=weights1, weights2=weights1, num_threads=num_threads)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result) == (N_jsamples+1, len(rp_bins), len(pi_bins)), msg

    grid_result = npairs_jackknife_xy_z(grid_points, grid_points, rp_bins, pi_bins, period=None,
        jtags1=grid_indices, jtags2=grid_indices, N_samples=grid_jackknife_ncells**3,
        num_threads=num_threads)

    for icell in range(1, grid_jackknife_ncells**3-1):
        assert np.all(grid_result[icell, :, :] == grid_result[icell+1, :, :])
