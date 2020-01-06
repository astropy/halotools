# cython: language_level=2
# cython: profile=False
"""
calculate and return the conditional pairwise distances between two sets of points.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys
cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from .distances cimport *
ctypedef bint (*f_type)(np.float64_t* w1, np.float64_t* w2)

__all__ = ['conditional_pairwise_distance_no_pbc',\
           'conditional_pairwise_xy_z_distance_no_pbc',]
__author__=['Duncan Campbell']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def conditional_pairwise_distance_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                             np.ndarray[np.float64_t, ndim=1] y_icell1,
                             np.ndarray[np.float64_t, ndim=1] z_icell1,
                             np.ndarray[np.float64_t, ndim=1] x_icell2,
                             np.ndarray[np.float64_t, ndim=1] y_icell2,
                             np.ndarray[np.float64_t, ndim=1] z_icell2,
                             np.float64_t max_r,
                             np.ndarray[np.float64_t, ndim=2] w_icell1,
                             np.ndarray[np.float64_t, ndim=2] w_icell2,
                             np.int_t cond_func_id):

    """
    Calculate the conditional limited pairwise distance matrix, :math:`d_{ij}`.

    calculate the distance between all pairs with sperations less than or equal
    to ``max_r`` if a conditon is met.

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

    w_icell1 : numpy.array
        array of floats

    w_icell2 : numpy.array
        array of floats

    cond_func_id : int
        integer ID of conditional function

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
    >>> weights = np.random.random(Npts)

    Calculate the distance between all pairs with separations less than 0.5 if
    the weight associated with the first point is larger than the second point:

    >>> r_max = 0.5
    >>> d,i,j = conditional_pairwise_distance_no_pbc(x,y,z,x,y,z,r_max,weights,weights,1)
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

    cond_func = return_conditional_function(cond_func_id)

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            if cond_func(&w_icell1[i,0],&w_icell2[j,0]):

                #calculate the square distance
                d = square_distance(x_icell1[i],y_icell1[i],z_icell1[i],\
                                x_icell2[j],y_icell2[j],z_icell2[j])

                #add distance to result
                if d<=max_r:
                    distances.push_back(d)
                    i_ind.push_back(i)
                    j_ind.push_back(j)
                    n = n+1

    return np.sqrt(distances).astype(float), np.array(i_ind).astype(int),\
           np.array(j_ind).astype(int)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def conditional_pairwise_xy_z_distance_no_pbc(np.ndarray[np.float64_t, ndim=1] x_icell1,
                                  np.ndarray[np.float64_t, ndim=1] y_icell1,
                                  np.ndarray[np.float64_t, ndim=1] z_icell1,
                                  np.ndarray[np.float64_t, ndim=1] x_icell2,
                                  np.ndarray[np.float64_t, ndim=1] y_icell2,
                                  np.ndarray[np.float64_t, ndim=1] z_icell2,
                                  np.float64_t max_rp, np.float64_t max_pi,
                                  np.ndarray[np.float64_t, ndim=2] w_icell1,
                                  np.ndarray[np.float64_t, ndim=2] w_icell2,
                                  np.int_t cond_func_id):
    """
    Calculate the conditional limited pairwise distance matrices, :math:`d_{{\perp}ij}` and :math:`d_{{\parallel}ij}`.

    Calculate the perpendicular and parallel distance between all pairs with separations
    less than or equal to ``max_rp`` and ``max_pi`` wrt to the z-direction, repsectively,
    if a conditon is met.

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

    w_icell1 : numpy.array
        array of floats

    w_icell2 : numpy.array
        array of floats

    cond_func_id : int
        integer ID of conditional function

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
    >>> weights = np.random.random(Npts)

    Calculate the distance between all pairs with perpednicular separations less than 0.25
    and parallel sperations of 0.5:

    >>> max_rp = 0.25
    >>> max_pi = 0.5
    >>> d_perp,d_para,i,j = conditional_pairwise_xy_z_distance_no_pbc(x,y,z,x,y,z,max_rp,max_para,weights,weights,1)
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

    cond_func = return_conditional_function(cond_func_id)

    #loop over points in grid1's cells
    n=0
    for i in range(0,Ni):
        #loop over points in grid2's cells
        for j in range(0,Nj):

            if cond_func(&w_icell1[i,0],&w_icell2[j,0]):

                #calculate the square distance
                d_perp = perp_square_distance(x_icell1[i], y_icell1[i],\
                                          x_icell2[j], y_icell2[j])
                d_para = para_square_distance(z_icell1[i], z_icell2[j])

                #add distance to result
                if (d_perp<=max_rp) & (d_para<=max_pi):
                    perp_distances.push_back(d_perp)
                    para_distances.push_back(d_para)
                    i_ind.push_back(i)
                    j_ind.push_back(j)
                    n = n+1

    return np.sqrt(perp_distances).astype(float), np.sqrt(para_distances).astype(float),\
           np.array(i_ind).astype(int), np.array(j_ind).astype(int)


### conditional functions ###
cdef bint gt_cond(np.float64_t* w1, np.float64_t* w2):
    """
    1
    """
    cdef bint result
    result = (w1[0]>w2[0])
    return result

cdef bint lt_cond(np.float64_t* w1, np.float64_t* w2):
    """
    2
    """
    cdef bint result
    result = (w1[0]<w2[0])
    return result

cdef bint eq_cond(np.float64_t* w1, np.float64_t* w2):
    """
    3
    """
    cdef bint result
    result = (w1[0]==w2[0])
    return result

cdef bint neq_cond(np.float64_t* w1, np.float64_t* w2):
    """
    4
    """
    cdef bint result
    result = (w1[0]!=w2[0])
    return result

cdef bint tg_cond(np.float64_t* w1, np.float64_t* w2):
    """
    5
    """
    cdef bint result
    result = (w1[0]>(w2[0]+w1[1]))
    return result

cdef bint lg_cond(np.float64_t* w1, np.float64_t* w2):
    """
    6
    """
    cdef bint result
    result = (w1[0]<(w2[0]+w1[1]))
    return result

cdef f_type return_conditional_function(cond_func_id):
    """
    returns a pointer to the user-specified conditional function.
    """

    if cond_func_id==1:
        return gt_cond
    if cond_func_id==2:
        return lt_cond
    if cond_func_id==3:
        return eq_cond
    if cond_func_id==4:
        return neq_cond
    if cond_func_id==5:
        return tg_cond
    if cond_func_id==6:
        return lg_cond
    else:
        raise ValueError('conditonal function does not exist!')
