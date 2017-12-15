"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import random_correlation
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import _principal_axes_from_inertia_tensors, inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh


__all__ = ('test_inertia_tensor1', )
fixed_seed = 43


def pure_python_inertia_tensor(m, x, y, z, p0=(0, 0, 0)):
    data = np.zeros((3, 3))
    dx = x - p0[0]
    dy = y - p0[1]
    dz = z - p0[2]

    data[0, 0] = np.sum(m*dx*dx)
    data[1, 1] = np.sum(m*dy*dy)
    data[2, 2] = np.sum(m*dz*dz)

    data[0, 1] = np.sum(m*dx*dy)
    data[1, 0] = data[0, 1]

    data[0, 2] = np.sum(m*dx*dz)
    data[2, 0] = data[0, 2]

    data[1, 2] = np.sum(m*dy*dz)
    data[2, 1] = data[1, 2]

    return data/np.sum(m)


def test_principal_axes_from_inertia_tensors1():
    """ Starting from 500 random positive definite symmetric matrices,
    enforce that the axes returned by the _principal_axes_from_inertia_tensors function
    are actually eigenvectors with the correct eigenvalue.
    """
    npts = int(500)
    correct_evals = (0.3, 0.7, 2.0)
    matrices = np.array([random_correlation.rvs(correct_evals) for __ in range(npts)])
    assert matrices.shape == (npts, 3, 3)

    principal_axes, evals = _principal_axes_from_inertia_tensors(matrices)

    assert np.shape(principal_axes) == (npts, 3)
    assert np.shape(evals) == (npts, )
    assert np.allclose(evals, correct_evals[2])

    for i in range(npts):
        m = matrices[i, :, :]
        x = principal_axes[i, 0]
        y = principal_axes[i, 1]
        z = principal_axes[i, 2]
        p = np.array((x, y, z))
        q = np.matmul(m, p)
        assert np.allclose(q, correct_evals[2]*p)


def test_inertia_tensor1():
    """ Calculate the inertia for sample1 on a regular grid and sample2
    a tight collection of point masses surrounding each sample1 point.
    Enforce that the returned tensor collection has the correct shape,
    and that each tensor is symmetric and positive definite.
    """

    Lbox, rsmooth = 250., 5.
    pos1 = generate_3d_regular_mesh(5, 0, Lbox)
    npts2_per_point = 5

    small_sphere_size = rsmooth/100.

    pos2 = generate_locus_of_3d_points(npts2_per_point,
                xc=pos1[0, 0], yc=pos1[0, 1], zc=pos1[0, 2],
                epsilon=small_sphere_size, seed=fixed_seed)
    for i in range(1, pos1.shape[0]):
        pos2_i = generate_locus_of_3d_points(npts2_per_point,
                    xc=pos1[i, 0], yc=pos1[i, 1], zc=pos1[i, 2],
                    epsilon=small_sphere_size, seed=fixed_seed)
        pos2 = np.vstack((pos2, pos2_i))

    with NumpyRNGContext(fixed_seed):
        masses = np.random.random(pos2.shape[0])

    tensors = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=Lbox)
    assert tensors.shape == (pos1.shape[0], 3, 3)

    assert np.all(tensors[:, 0, 0] > 0)
    assert np.all(tensors[:, 1, 1] > 0)
    assert np.all(tensors[:, 2, 2] > 0)

    assert np.allclose(tensors[:, 0, 1], tensors[:, 1, 0])
    assert np.allclose(tensors[:, 0, 2], tensors[:, 2, 0])
    assert np.allclose(tensors[:, 1, 2], tensors[:, 2, 1])

    evals, evecs = np.linalg.eigh(tensors)
    assert np.all(evals > 0)


def test_inertia_tensor2():
    """ Calculate the inertia for sample1 on a regular grid and sample2
    a tight collection of point masses randomly placed to surround each sample1 point.
    For each point in sample1, enforce that the returned tensor agrees
    with the results from a pure python calculation
    """

    Lbox, rsmooth = 1., 0.05
    pos1 = generate_3d_regular_mesh(2, 0, Lbox)
    npts2_per_point = 5
    small_sphere_size = rsmooth/100.

    pos2 = generate_locus_of_3d_points(npts2_per_point,
                xc=pos1[0, 0], yc=pos1[0, 1], zc=pos1[0, 2],
                epsilon=small_sphere_size, seed=fixed_seed)
    for i in range(1, pos1.shape[0]):
        pos2_i = generate_locus_of_3d_points(npts2_per_point,
                    xc=pos1[i, 0], yc=pos1[i, 1], zc=pos1[i, 2],
                    epsilon=small_sphere_size, seed=fixed_seed)
        pos2 = np.vstack((pos2, pos2_i))

    with NumpyRNGContext(fixed_seed):
        masses = np.random.random(pos2.shape[0])

    tensors = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=None)
    npts1 = tensors.shape[0]
    for i in range(npts1):
        t = tensors[i, :, :]
        ifirst, ilast = i*npts2_per_point, (i+1)*npts2_per_point
        x = pos2[ifirst:ilast, 0]
        y = pos2[ifirst:ilast, 1]
        z = pos2[ifirst:ilast, 2]
        p0 = pos1[i, :]
        m = masses[ifirst:ilast]
        t_python = pure_python_inertia_tensor(m, x, y, z, p0=p0)
        assert np.allclose(t, t_python, rtol=1e-3)


def test_serial_parallel_agreement():
    Lbox, rsmooth = 1., 0.05
    pos1 = generate_3d_regular_mesh(3, 0, Lbox)
    npts2_per_point = 5
    small_sphere_size = rsmooth/100.

    pos2 = generate_locus_of_3d_points(npts2_per_point,
                xc=pos1[0, 0], yc=pos1[0, 1], zc=pos1[0, 2],
                epsilon=small_sphere_size, seed=fixed_seed)
    for i in range(1, pos1.shape[0]):
        pos2_i = generate_locus_of_3d_points(npts2_per_point,
                    xc=pos1[i, 0], yc=pos1[i, 1], zc=pos1[i, 2],
                    epsilon=small_sphere_size, seed=fixed_seed)
        pos2 = np.vstack((pos2, pos2_i))

    with NumpyRNGContext(fixed_seed):
        masses = np.random.random(pos2.shape[0])

    tensors_serial = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, num_threads=1)
    tensors_parallel = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, num_threads=2)
    assert np.shape(tensors_serial) == np.shape(tensors_parallel)
    assert np.allclose(tensors_serial, tensors_parallel, rtol=0.001)


