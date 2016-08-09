""" Module storing pure python brute force per-object pair-counters used for unit-testing.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = ('pure_python_npairs_per_object_3d', )


def pure_python_npairs_per_object_3d(sample1, sample2, rbins, period=None):
    """
    """
    if period is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    else:
        xperiod, yperiod, zperiod = period, period, period

    npts1, npts2, num_rbins = len(sample1), len(sample2), len(rbins)

    counts = np.zeros((npts1, num_rbins), dtype=int)

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

            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            for irbin, r in enumerate(rbins):
                if d < r:
                    counts[i, irbin] += 1

    return counts
