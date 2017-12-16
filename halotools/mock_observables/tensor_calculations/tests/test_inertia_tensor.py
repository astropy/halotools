"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh


__all__ = ('test_inertia_tensor1', )
fixed_seed = 43


def pure_python_inertia_tensor(m, x, y, z, p0=(0, 0, 0)):
    """ Numpy elementwise calculation of the inertia tensor for a single
    collection of massive points.
    """
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

    return data, np.sum(m)


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

    tensors, sum_of_masses = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=Lbox)
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
    For each point in sample1, enforce that each of the returned tensors agrees
    with the results of an independent implementation that uses
    numpy instead of the cython kernels
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

    tensors, sum_m_cython = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=None)
    npts1 = tensors.shape[0]
    for i in range(npts1):
        t = tensors[i, :, :]
        ifirst, ilast = i*npts2_per_point, (i+1)*npts2_per_point
        x = pos2[ifirst:ilast, 0]
        y = pos2[ifirst:ilast, 1]
        z = pos2[ifirst:ilast, 2]
        p0 = pos1[i, :]
        m = masses[ifirst:ilast]
        t_python, sum_m_python = pure_python_inertia_tensor(m, x, y, z, p0=p0)
        assert np.allclose(t, t_python, rtol=1e-3)


def test_serial_parallel_agreement():
    """ Distribute massive points in tight spheres surrounding a grid of `sample1` points,
    and verify that the returned inertia tensors are agree when the function is
    called in serial or parallel.
    """
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

    tensors_serial, sum_m_serial = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, num_threads=1)
    tensors_parallel, sum_m_parallel = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, num_threads=2)
    assert np.shape(tensors_serial) == np.shape(tensors_parallel)
    assert np.allclose(tensors_serial, tensors_parallel, rtol=0.001)


def test_pbcs():
    """ Construct a test case where PBCs should not matter, and enforce that the function returns
    the same inertia matrices regardless of whether an input period is passed
    """
    Lbox, rsmooth = 1., 0.01
    pos1 = generate_3d_regular_mesh(3, 0, Lbox)
    npts2_per_point = 5
    small_sphere_size = rsmooth/10.

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

    tensors1, sum_m_cython1 = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=None)
    tensors2, sum_m_cython2 = inertia_tensor_per_object(pos1, pos2, masses, rsmooth, period=Lbox)
    assert np.allclose(tensors1, tensors2)
    assert np.allclose(sum_m_cython1, sum_m_cython2)
