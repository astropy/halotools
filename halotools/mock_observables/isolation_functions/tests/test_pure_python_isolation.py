""" Module providing testing for the brute force isolation functions
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_isolation import naive_spherical_isolation

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test_naive_spherical_isolation1', )

fixed_seed = 43


def test_naive_spherical_isolation1():
    npts1, npts2 = 50, 45
    with NumpyRNGContext(fixed_seed):
        x1arr = np.random.rand(npts1)
        y1arr = np.random.rand(npts1)
        z1arr = np.random.rand(npts1)
        x2arr = np.random.rand(npts2)
        y2arr = np.random.rand(npts2)
        z2arr = np.random.rand(npts2)

    rmax = 0.25
    xperiod, yperiod, zperiod = 1, 1, 1
    result = naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rmax, xperiod, yperiod, zperiod)
