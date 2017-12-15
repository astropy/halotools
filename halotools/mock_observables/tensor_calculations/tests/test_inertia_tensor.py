"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import random_correlation
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import _principal_axes_from_matrices, inertia_tensor_per_object

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh


__all__ = ('test_inertia_tensor1', )
fixed_seed = 43


def test_principal_axes_from_matrices1():
    """ Starting from 500 random positive definite symmetric matrices,
    enforce that the axes returned by the _principal_axes_from_matrices function
    are actually eigenvectors with the correct eigenvalue.
    """
    npts = int(500)
    correct_evals = (0.3, 0.7, 2.0)
    matrices = np.array([random_correlation.rvs(correct_evals) for __ in range(npts)])
    assert matrices.shape == (npts, 3, 3)

    principal_axes, evals = _principal_axes_from_matrices(matrices)

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

    assert np.any(tensors > 0)


