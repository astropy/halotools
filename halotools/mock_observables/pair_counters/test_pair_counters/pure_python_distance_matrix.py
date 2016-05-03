""" Module storing pure python brute force pair matrices used for unit-testing. 
"""

from __future__ import absolute_import, division, print_function, unicode_literals
    
import numpy as np

__all__ = ('pure_python_distance_matrix_3d', )

def pure_python_distance_matrix_3d(
    sample1, sample2, velocities2, rmin, rmax, Lbox = None):
    """ Brute force pure python function calculating the distance 
    between all pairs of points and storing the result into a matrix, 
    accounting for possible periodicity of the box.
    """ 
    if Lbox is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf 
    else:
        xperiod, yperiod, zperiod = Lbox, Lbox, Lbox 

    npts1, npts2 = len(sample1), len(sample2)

    pair_matrix = np.zeros((npts1, npts2))

    for i in range(npts1):
        for j in range(npts2):
            dx = sample1[i,0] - sample2[j,0]
            dy = sample1[i,1] - sample2[j,1]
            dz = sample1[i,2] - sample2[j,2]

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

            pair_matrix[i, j] = np.sqrt(dx*dx + dy*dy + dz*dz)

    return pair_matrix

def pure_python_distance_matrix_xy_z(
    sample1, sample2, velocities2, rmin, rmax, Lbox = None):
    """ Brute force pure python function calculating the distance 
    between all pairs of points and storing the result into two matrices, 
    one storing xy-distances, the other storing z-distances, 
    account for possible periodicity of the box.
    """ 
    if Lbox is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf 
    else:
        xperiod, yperiod, zperiod = Lbox, Lbox, Lbox 

    npts1, npts2 = len(sample1), len(sample2)

    pair_matrix_xy = np.zeros((npts1, npts2))
    pair_matrix_z = np.zeros((npts1, npts2))

    for i in range(npts1):
        for j in range(npts2):
            dx = sample1[i,0] - sample2[j,0]
            dy = sample1[i,1] - sample2[j,1]
            dz = sample1[i,2] - sample2[j,2]

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

            pair_matrix_xy[i, j] = np.sqrt(dx*dx + dy*dy)
            pair_matrix_z[i, j] = abs(dz)

    return pair_matrix_xy, pair_matrix_z

