"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = ('pure_python_counts_in_cylinders', 'pure_python_idx_in_cylinders')

fixed_seed = 43

def pure_python_idx_in_cylinders(
        sample1, sample2, rp_max, pi_max, period=None):
    if period is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    else:
        xperiod, yperiod, zperiod = period, period, period

    autocorr = False
    if sample2 is None:
        sample2, autocorr = sample1, True

    npts1, npts2 = len(sample1), len(sample2)

    indexes = []

    for i in range(npts1):
        for j in range(npts2):
            if i == j and autocorr:
                continue
            if _point_within_cylinder(sample1[i], sample2[j], rp_max[i], pi_max[i], xperiod, yperiod, zperiod):
                indexes.append((i, j))

    # https://github.com/numpy/numpy/issues/2407 for str("colname")
    return np.array(indexes, dtype=[(str("i1"), np.uint32), (str("i2"), np.uint32)])


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

    autocorr = False
    if sample2 is None:
        sample2, autocorr = sample1, True

    npts1, npts2 = len(sample1), len(sample2)

    counts = np.zeros(npts1, dtype=int)

    for i in range(npts1):
        for j in range(npts2):
            if i == j and autocorr:
                continue
            if _point_within_cylinder(sample1[i], sample2[j], rp_max[i], pi_max[i], xperiod, yperiod, zperiod):
                counts[i] += 1
    return counts

def _point_within_cylinder(p1, p2, rp_cylinder, pi_cylinder, xperiod, yperiod, zperiod):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]

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
    return (rp <= rp_cylinder) & (dz <= pi_cylinder)
