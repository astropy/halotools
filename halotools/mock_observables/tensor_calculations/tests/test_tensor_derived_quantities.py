"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import random_correlation
from astropy.utils.misc import NumpyRNGContext

from ..inertia_tensor import inertia_tensor_per_object_3d
from ..tensor_derived_quantities import principal_axes_from_inertia_tensors

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
