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
    npts2 = 45
    sample1 = generate_3d_regular_mesh(5)  #  0.1, 0.3, 0.5, 0.7, 0.9
    npts1 = sample1.shape[0]
    sample2 = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.3, zc=0.3)
    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]

    rmax = 0.02
    xperiod, yperiod, zperiod = 1, 1, 1
    result = naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rmax, xperiod, yperiod, zperiod)
    assert result.sum() == npts1 - 1


def test_naive_spherical_isolation2():
    npts2 = 45
    sample1 = generate_3d_regular_mesh(5)  #  0.1, 0.3, 0.5, 0.7, 0.9
    npts1 = sample1.shape[0]
    sample2a = generate_locus_of_3d_points(npts2, xc=0.101, yc=0.3, zc=0.3)
    sample2b = generate_locus_of_3d_points(npts2, xc=0.301, yc=0.3, zc=0.3)
    sample2c = generate_locus_of_3d_points(npts2, xc=0.501, yc=0.3, zc=0.3)
    sample2 = np.concatenate((sample2a, sample2b, sample2c))
    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]

    rmax = 0.02
    xperiod, yperiod, zperiod = 1, 1, 1
    result = naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rmax, xperiod, yperiod, zperiod)
    assert result.sum() == npts1 - 3
