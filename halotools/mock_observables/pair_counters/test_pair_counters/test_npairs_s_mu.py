#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

# load pair counters
from ..npairs_s_mu import npairs_s_mu
from ....mock_observables import npairs_3d
# load comparison simple pair counters

from astropy.tests.helper import pytest
slow = pytest.mark.slow

__all__ = ('test_npairs_s_mu_periodic', 'test_npairs_s_mu_nonperiodic')

# set up random points to test pair counters
np.random.seed(1)
Npts = 1000
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


def test_npairs_s_mu_periodic():
    """
    test npairs_s_mu with periodic boundary conditions.
    """

    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    result = npairs_s_mu(random_sample, random_sample, s_bins, mu_bins,
        period=period, num_threads=num_threads)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins), N_mu_bins), msg

    result = np.diff(result, axis=1)
    result = np.sum(result, axis=1)+ Npts

    test_result = npairs_3d(random_sample, random_sample, s_bins,
        period=period, num_threads=num_threads)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg


def test_npairs_s_mu_nonperiodic():
    """
    test npairs_s_mu without periodic boundary conditions.
    """

    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins=100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)

    result = npairs_s_mu(random_sample, random_sample, s_bins, mu_bins,
        num_threads=num_threads)

    msg = 'The returned result is an unexpected shape.'
    assert np.shape(result)==(len(s_bins), N_mu_bins), msg

    result = np.diff(result, axis=1)
    result = np.sum(result, axis=1)+ Npts

    test_result = npairs_3d(random_sample, random_sample, s_bins, num_threads=num_threads)

    msg = "The double tree's result(s) are not equivalent to simple pair counter's."
    assert np.all(result == test_result), msg
