"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.utils.misc import NumpyRNGContext
import pytest

from .pure_python_distance_matrix import pure_python_distance_matrix_3d, pure_python_distance_matrix_xy_z

from ..pairwise_distance_3d import pairwise_distance_3d, _get_r_max
from ..pairwise_distance_elliptical import pairwise_distance_elliptical

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

fixed_seed = 43


def test_1():
    """
    make sure the elliptical result is the same as the 3D result when the axis ratios are 1
    """

    Npts = 100
    with NumpyRNGContext(fixed_seed):
        random_sample_1 = np.random.random((Npts, 3))
        random_sample_2 = np.random.random((Npts, 3))
    period = 1.0
    q1 = 1.0
    s1 = 1.0
    ed_max = 0.1
    r_max = ed_max

    result_1 = pairwise_distance_elliptical(random_sample_1, q1, s1, random_sample_2, ed_max)
    result_2 = pairwise_distance_3d(random_sample_1, random_sample_2, r_max)

    assert (result_1!=result_2).nnz==0


