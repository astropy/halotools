""" Pure python functions to aid in unit-testing
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = ('pure_python_weighted_npairs_xy', 'periodic_xy_distance')


def periodic_xy_distance(x1, y1, x2, y2, xperiod, yperiod):
    """ Compute the distance between two points in the xy-direction,
    accounting for periodic boundary conditions.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > xperiod/2.:
        dx = xperiod - dx
    if dy > yperiod/2.:
        dy = yperiod - dy
    return np.sqrt(dx*dx + dy*dy)


def pure_python_weighted_npairs_xy(xarr1, yarr1, xarr2, yarr2, warr2, rp_bins, xperiod, yperiod):
    """ Count the number of pairs as a function of xy-distance,
    weighted by the second pair of points.
    """
    num_rp_bins = len(rp_bins)
    counts = np.zeros(num_rp_bins)
    weighted_counts = np.zeros(num_rp_bins)

    for x1, y1 in zip(xarr1, yarr1):
        for x2, y2, w2 in zip(xarr2, yarr2, warr2):
            rp_ij = periodic_xy_distance(x1, y1, x2, y2, xperiod, yperiod)
            for k, rp_bin_edge in enumerate(rp_bins):
                if rp_ij < rp_bin_edge:
                    counts[k] += 1
                    weighted_counts[k] += w2
    return counts, weighted_counts


