""" Module storing pure python brute force per-object pair-counters used for unit-testing.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .pure_python_weighted_npairs_xy import periodic_xy_distance

__all__ = ('pure_python_weighted_npairs_per_object_xy', )


def pure_python_weighted_npairs_per_object_xy(xarr1, yarr1, xarr2, yarr2, warr2,
        rp_bins, xperiod, yperiod):
    """ Count the number of pairs as a function of xy-distance,
    weighted by the second pair of points.
    """
    num_rp_bins = len(rp_bins)
    num_pts1 = len(xarr1)

    counts = np.zeros((num_pts1, num_rp_bins))
    weighted_counts = np.zeros((num_pts1, num_rp_bins))

    for i, x1, y1 in zip(range(num_pts1), xarr1, yarr1):
        for x2, y2, w2 in zip(xarr2, yarr2, warr2):
            rp_ij = periodic_xy_distance(x1, y1, x2, y2, xperiod, yperiod)
            for k, rp_bin_edge in enumerate(rp_bins):
                if rp_ij < rp_bin_edge:
                    counts[i, k] += 1
                    weighted_counts[i, k] += w2
    return counts, weighted_counts
