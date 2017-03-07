"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .pure_python_weighted_npairs_xy import pure_python_weighted_npairs_xy

from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ('test_pure_python_weighted_npairs_xy', )

fixed_seed = 43


def test_pure_python_weighted_npairs_xy():
    """
    """
    npts_per_dim = 5
    data1 = generate_3d_regular_mesh(npts_per_dim)
    data2 = generate_3d_regular_mesh(npts_per_dim)
    w2 = np.zeros(data2.shape[0]) + 2.5
    rp_bins = np.array((0.199, 0.201, 0.3))
    xperiod, yperiod = 1, 1

    xarr1, yarr1 = data1[:, 0], data1[:, 1]
    xarr2, yarr2 = data2[:, 0], data2[:, 1]
    counts, weighted_counts = pure_python_weighted_npairs_xy(
            xarr1, yarr1, xarr2, yarr2, w2, rp_bins, xperiod, yperiod)

    npts1_total = npts_per_dim**3

    assert counts[0] == npts1_total*npts_per_dim
    assert counts[1] == (npts1_total*5*npts_per_dim)
    assert counts[2] == (npts1_total*9*npts_per_dim)

    assert np.all(weighted_counts == 2.5*counts)

