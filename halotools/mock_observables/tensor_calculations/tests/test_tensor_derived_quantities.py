"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import random_correlation
from astropy.utils.misc import NumpyRNGContext

from ..tensor_derived_quantities import (
            principal_axes_from_inertia_tensors, sphericity_from_inertia_tensors)

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh


__all__ = ('test_principal_axes_from_inertia_tensors1', )
fixed_seed = 43


def test_principal_axes_from_inertia_tensors1():
    """ Starting from 500 random positive definite symmetric matrices,
    enforce that the axes returned by the _principal_axes_from_inertia_tensors function
    are actually eigenvectors with the correct eigenvalue.
    """
    npts = int(500)
    correct_evals = (0.3, 0.7, 2.0)
    matrices = np.array([random_correlation.rvs(correct_evals) for __ in range(npts)])
    assert matrices.shape == (npts, 3, 3)

    principal_axes, evals = principal_axes_from_inertia_tensors(matrices)

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


def test_sphericity_from_inertia_tensors():
    """ Use `scipy.stats.random_correlation` to generate matrices with known
    eigenvalues. Call the `sphericity_from_inertia_tensors` function to operate
    on these matrices, and verify that the returned sphericity agrees with
    the sphericity of the input eigenvalues used to define the matrices.
    """
    spherical_evals = (1., 1., 1.)
    non_spherical_evals = (0.1, 0.9, 2.)

    tensors = []
    tensors.append(random_correlation.rvs(spherical_evals))
    tensors.append(random_correlation.rvs(non_spherical_evals))
    tensors.append(random_correlation.rvs(non_spherical_evals))
    tensors.append(random_correlation.rvs(spherical_evals))
    matrices = np.array(tensors)

    sphericity = sphericity_from_inertia_tensors(matrices)
    assert np.allclose(sphericity, (1, 0.05, 0.05, 1))

