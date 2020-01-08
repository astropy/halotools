"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

# load pair counters
from ..npairs_jackknife_xy_z import npairs_jackknife_xy_z
from ..npairs_jackknife_xy_z import _npairs_jackknife_xy_z_process_weights_jtags as process_weights

from ....custom_exceptions import HalotoolsError

# load comparison simple pair counters

import pytest
from astropy.utils.misc import NumpyRNGContext

slow = pytest.mark.slow

__all__ = ('test_npairs_jackknife_xy_z_periodic','test_npairs_jackknife_xy_z_nonperiodic',
           'test_process_weights1','test_process_weights2','test_process_weights3',
           'test_process_weights4','test_process_weights5','test_process_weights6',
           'test_process_weights7','test_process_weights8','test_process_weights9')

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


def test_npairs_jackknife_xy_z_periodic():
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


def test_parallel_serial_consistency():
    """
    test npairs_jackknife_xy_z gives the same result with and w/o threading
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

    result_1 = npairs_jackknife_xy_z(random_sample, random_sample, rp_bins, pi_bins, period=period,
        jtags1=jtags1, jtags2=jtags1, N_samples=10,
        weights1=weights1, weights2=weights1, num_threads=1)

    result_2 = npairs_jackknife_xy_z(random_sample, random_sample, rp_bins, pi_bins, period=period,
        jtags1=jtags1, jtags2=jtags1, N_samples=10,
        weights1=weights1, weights2=weights1, num_threads=2)

    assert np.allclose(result_1,result_2), 'threaded and non-threaded results are not consistent'


def test_process_weights1():

    npts1, npts2 = 10000, 20000
    N_samples = 3
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, N_samples+1, npts1)
        jtags2 = np.random.randint(1, N_samples+1, npts2)

    __ = process_weights(sample1, sample2,
        weights1, weights2, jtags1, jtags2, N_samples)


def test_process_weights2():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1-1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, N_samples, npts1)
        jtags2 = np.random.randint(1, N_samples, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "weights1 should have same len as sample1"
    assert substr in err.value.args[0]


def test_process_weights3():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2-1)
        jtags1 = np.random.randint(1, N_samples, npts1)
        jtags2 = np.random.randint(1, N_samples, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "weights2 should have same len as sample2"
    assert substr in err.value.args[0]


def test_process_weights4():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, N_samples, npts1-1)
        jtags2 = np.random.randint(1, N_samples, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags1 should have same len as sample1"
    assert substr in err.value.args[0]


def test_process_weights5():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, N_samples, npts1)
        jtags2 = np.random.randint(1, N_samples, npts2-1)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags2 should have same len as sample2"
    assert substr in err.value.args[0]


def test_process_weights6():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(0, 2, npts1)
        jtags2 = np.random.randint(1, 2, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags1 must be >= 1"
    assert substr in err.value.args[0]


def test_process_weights7():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, 2, npts1)
        jtags2 = np.random.randint(0, 2, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags2 must be >= 1"
    assert substr in err.value.args[0]


def test_process_weights8():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(N_samples+1, N_samples+10, npts1)
        jtags2 = np.random.randint(1, 2, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags1 must be <= N_samples"
    assert substr in err.value.args[0]


def test_process_weights9():

    npts1, npts2 = 10, 10
    N_samples = 5
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))
        weights1 = np.random.rand(npts1)
        weights2 = np.random.rand(npts2)
        jtags1 = np.random.randint(1, 2, npts1)
        jtags2 = np.random.randint(N_samples+1, N_samples+10, npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = process_weights(sample1, sample2,
            weights1, weights2, jtags1, jtags2, N_samples)
    substr = "jtags2 must be <= N_samples"
    assert substr in err.value.args[0]
