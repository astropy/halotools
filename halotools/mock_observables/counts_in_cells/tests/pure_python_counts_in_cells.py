"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('pure_python_counts_in_cylinders', )

fixed_seed = 43


def pure_python_counts_in_cylinders(
        sample1, sample2, rp_max, pi_max, period=None):
    """ Brute force pure python function calculating the distance
    between all pairs of points and storing the result into two matrices,
    one storing xy-distances, the other storing z-distances,
    account for possible periodicity of the box.
    """
    if period is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    else:
        xperiod, yperiod, zperiod = period, period, period

    npts1, npts2 = len(sample1), len(sample2)

    counts = np.zeros(npts1, dtype=int)

    for i in range(npts1):
        for j in range(npts2):
            dx = sample1[i, 0] - sample2[j, 0]
            dy = sample1[i, 1] - sample2[j, 1]
            dz = sample1[i, 2] - sample2[j, 2]

            if dx > xperiod/2.:
                dx = xperiod - dx
            elif dx < -xperiod/2.:
                dx = -(xperiod + dx)

            if dy > yperiod/2.:
                dy = yperiod - dy
            elif dy < -yperiod/2.:
                dy = -(yperiod + dy)

            if dz > zperiod/2.:
                dz = zperiod - dz
            elif dz < -zperiod/2.:
                dz = -(zperiod + dz)

            rp = np.sqrt(dx*dx + dy*dy)
            dz = abs(dz)
            rp_cylinder = rp_max[i]
            pi_cylinder = pi_max[i]
            if (rp <= rp_cylinder) & (dz <= pi_cylinder):
                counts[i] += 1

    return counts
