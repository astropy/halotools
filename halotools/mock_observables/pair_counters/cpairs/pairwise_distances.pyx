# cython: language_level=2
# cython: profile=False
"""
calculate and return the pairwise distances between two sets of points.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from .distances cimport *

__all__ = ('pairwise_distance_no_pbc', 'pairwise_distance_pbc',
    'pairwise_xy_z_distance_no_pbc', 'pairwise_xy_z_distance_pbc')
__author__=['Duncan Campbell']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.float64_t max_r):

    r"""
    Calculate the limited pairwise distance matrix, :math:`d_{ij}`.

    calculate the distance between all pairs with sperations less than or equal
    to ``max_r``.

    Parameters
    ----------
    x_icell1 : numpy.array
        array of x positions of length N1 (data1)

    y_icell1 : numpy.array
        array of y positions of length N1 (data1)

    z_icell1 : numpy.array
        array of z positions of length N1 (data1)

    x_icell2 : numpy.array
        array of x positions of length N2 (data2)

    y_icell2 : numpy.array
        array of y positions of length N2 (data2)

    z_icell2 : numpy.array
        array of z positions of length N2 (data2)

    max_r : float
        maximum separation to record

    Returns
    -------
    d : numpy.array
        array of pairwise separation distances

    i : numpy.array
        array of 0-indexed indices

    j : numpy.array
        array of 0-indexed indices

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    unit cube.

    >>> Npts = 1000

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    Calculate the distance between all pairs with separations less than 0.5:

    >>> r_max = 0.5
    >>> d,i,j = pairwise_distance_no_pbc(x,y,z,x,y,z,r_max)
    """

    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] distances
    cdef double d
    cdef np.int_t i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)

    #square the distance to avoid taking a square root in a tight loop
    max_r = max_r**2

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            #calculate the square distance
            d = square_distance(
                x_icell1[i],y_icell1[i],z_icell1[i],
                x_icell2[j],y_icell2[j],z_icell2[j])

            #add distance to result
            if d<=max_r:
                distances.push_back(d)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1

    return (np.sqrt(distances).astype(float),
        np.array(i_ind).astype(int),np.array(j_ind).astype(int))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_distance_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.ndarray[np.float64_t, ndim=1] period,
                             np.float64_t max_r):

    """
    Calculate the limited pairwise distance matrix, :math:`d_{ij}`, with periodic boundary conditions (PBC).

    calculate the distance between all pairs with sperations less than or equal
    to ``max_r``.

    Parameters
    ----------
    x_icell1 : numpy.array
        array of x positions of length N1 (data1)

    y_icell1 : numpy.array
        array of y positions of length N1 (data1)

    z_icell1 : numpy.array
        array of z positions of length N1 (data1)

    x_icell2 : numpy.array
        array of x positions of length N2 (data2)

    y_icell2 : numpy.array
        array of y positions of length N2 (data2)

    z_icell2 : numpy.array
        array of z positions of length N2 (data2)

    period : numpy.array
        array defining  periodic boundary conditions.

    max_r : float
        maximum separation to record

    Returns
    -------
    d : numpy.array
        array of pairwise separation distances

    i : numpy.array
        array of 0-indexed indices

    j : numpy.array
        array of 0-indexed indices

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    Calculate the distance between all pairs with separations less than 0.5:

    >>> r_max = 0.5
    >>> d,i,j = pairwise_distance_pbc(x,y,z,x,y,z,period,r_max)
    """

    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] distances
    cdef double d
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)

    #square the distance to avoid taking a square root in a tight loop
    max_r = max_r**2

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            #calculate the square distance
            d = periodic_square_distance(
                x_icell1[i],y_icell1[i],z_icell1[i],
                x_icell2[j],y_icell2[j],z_icell2[j],
                <np.float64_t*> period.data)

            #add distance to result
            if d<=max_r:
                distances.push_back(d)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1

    return (np.sqrt(distances).astype(float),
        np.array(i_ind).astype(int), np.array(j_ind).astype(int))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_xy_z_distance_no_pbc(
    np.ndarray[np.float64_t, ndim=1] x_icell1,
    np.ndarray[np.float64_t, ndim=1] y_icell1,
    np.ndarray[np.float64_t, ndim=1] z_icell1,
    np.ndarray[np.float64_t, ndim=1] x_icell2,
    np.ndarray[np.float64_t, ndim=1] y_icell2,
    np.ndarray[np.float64_t, ndim=1] z_icell2,
    np.float64_t max_rp, np.float64_t max_pi):
    """
    Calculate the limited pairwise distance matrices, :math:`d_{{\perp}ij}` and :math:`d_{{\parallel}ij}`.

    Calculate the perpendicular and parallel distance between all pairs with separations
    less than or equal to ``max_rp`` and ``max_pi`` wrt to the z-direction, repsectively.

    Parameters
    ----------
    x_icell1 : numpy.array
        array of x positions of length N1 (data1)

    y_icell1 : numpy.array
        array of y positions of length N1 (data1)

    z_icell1 : numpy.array
        array of z positions of length N1 (data1)

    x_icell2 : numpy.array
        array of x positions of length N2 (data2)

    y_icell2 : numpy.array
        array of y positions of length N2 (data2)

    z_icell2 : numpy.array
        array of z positions of length N2 (data2)

    max_rp : float
        maximum perpendicular separation to record

    max_pi : float
        maximum parallel separation to record

    Returns
    -------
    d_perp : numpy.array
        array of perpendicular pairwise separation distances

    d_para : numpy.array
        array of parallel pairwise separation distances

    i : numpy.array
        array of 0-indexed indices

    j : numpy.array
        array of 0-indexed indices

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    unit cube.

    >>> Npts = 1000

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    Calculate the distance between all pairs with perpednicular separations less than 0.25
    and parallel sperations of 0.5:

    >>> max_rp = 0.25
    >>> max_pi = 0.5
    >>> d_perp,d_para,i,j = pairwise_xy_z_distance_no_pbc(x,y,z,x,y,z,max_rp,max_para)
    """


    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] para_distances
    cdef vector[np.float64_t] perp_distances
    cdef double d_perp, d_para
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)

    #square the distance bins to avoid taking a square root in a tight loop
    max_rp = max_rp**2
    max_pi = max_pi**2

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            #calculate the square distance
            d_perp = perp_square_distance(
                x_icell1[i], y_icell1[i],
                x_icell2[j], y_icell2[j])
            d_para = para_square_distance(z_icell1[i], z_icell2[j])

            #add distance to result
            if (d_perp<=max_rp) & (d_para<=max_pi):
                perp_distances.push_back(d_perp)
                para_distances.push_back(d_para)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1

    return (np.sqrt(perp_distances).astype(float),
        np.sqrt(para_distances).astype(float),
        np.array(i_ind).astype(int), np.array(j_ind).astype(int))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def pairwise_xy_z_distance_pbc(
    np.ndarray[np.float64_t, ndim=1] x_icell1,
    np.ndarray[np.float64_t, ndim=1] y_icell1,
    np.ndarray[np.float64_t, ndim=1] z_icell1,
    np.ndarray[np.float64_t, ndim=1] x_icell2,
    np.ndarray[np.float64_t, ndim=1] y_icell2,
    np.ndarray[np.float64_t, ndim=1] z_icell2,
    np.ndarray[np.float64_t, ndim=1] period,
    np.float64_t max_rp, np.float64_t max_pi):

    """
    Calculate the limited pairwise distance matrices, :math:`d_{{\perp}ij}` and :math:`d_{{\parallel}ij}`, with periodic boundary conditions (PBC).

    Calculate the perpendicular and parallel distance between all pairs with separations
    less than or equal to ``max_rp`` and ``max_pi`` wrt to the z-direction, repsectively.

    Parameters
    ----------
    x_icell1 : numpy.array
        array of x positions of length N1 (data1)

    y_icell1 : numpy.array
        array of y positions of length N1 (data1)

    z_icell1 : numpy.array
        array of z positions of length N1 (data1)

    x_icell2 : numpy.array
        array of x positions of length N2 (data2)

    y_icell2 : numpy.array
        array of y positions of length N2 (data2)

    z_icell2 : numpy.array
        array of z positions of length N2 (data2)

    period : numpy.array
        array defining  periodic boundary conditions.

    max_rp : float
        maximum perpendicular separation to record

    max_pi : float
        maximum parallel separation to record

    Returns
    -------
    d_perp : numpy.array
        array of perpendicular pairwise separation distances

    d_para : numpy.array
        array of parallel pairwise separation distances

    i : numpy.array
        array of 0-indexed indices

    j : numpy.array
        array of 0-indexed indices

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic unit cube.

    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])

    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)

    Calculate the distance between all pairs with perpednicular separations less than 0.25
    and parallel sperations of 0.5:

    >>> max_rp = 0.25
    >>> max_pi = 0.5
    >>> d_perp, d_para,i,j = pairwise_xy_z_distance_no_pbc(x,y,z,x,y,z,period,max_rp,max_para)
    """

    #c definitions
    cdef vector[np.int_t] i_ind
    cdef vector[np.int_t] j_ind
    cdef vector[np.float64_t] para_distances
    cdef vector[np.float64_t] perp_distances
    cdef double d_perp, d_para
    cdef int i, j, n
    cdef int Ni = len(x_icell1)
    cdef int Nj = len(x_icell2)

    #square the distance bins to avoid taking a square root in a tight loop
    max_rp = max_rp**2
    max_pi = max_pi**2

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            #calculate the square distance
            d_perp = periodic_perp_square_distance(
                x_icell1[i],y_icell1[i],
                x_icell2[j],y_icell2[j],
                <np.float64_t*>period.data)
            d_para = periodic_para_square_distance(
                z_icell1[i], z_icell2[j],<np.float64_t*>period.data)

            #add distance to result
            if (d_perp<=max_rp) & (d_para<=max_pi):
                perp_distances.push_back(d_perp)
                para_distances.push_back(d_para)
                i_ind.push_back(i)
                j_ind.push_back(j)
                n = n+1

    return (np.sqrt(perp_distances).astype(float),
        np.sqrt(para_distances).astype(float),
        np.array(i_ind).astype(int), np.array(j_ind).astype(int))


