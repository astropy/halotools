# -*- coding: utf-8 -*-

"""
simple python brute force pair counting functions.  The primary purpose of these functions
is as a sanity check on more complex pair counting techniques.  These functions should not
be used on large data sets, as memory usage is very large, and runtimes can be very slow.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

__all__=['npairs', 'wnpairs', 'xy_z_npairs', 'xy_z_wnpairs', 'pairs']
__author__ = ['Duncan Campbell']


def npairs(sample1, sample2, rbins, period=None):
    """
    Calculate the number of pairs with separations less than or equal to rbins[i].

    Parameters
    ----------
    sample1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    sample2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted.
        len(rbins) = Nrbins + 1.

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs

    """

    #work with arrays!
    sample1 = np.asarray(sample1)
    if sample1.ndim ==1: sample1 = np.array([sample1])
    sample2 = np.asarray(sample2)
    if sample2.ndim ==1: sample2 = np.array([sample2])
    rbins = np.asarray(rbins)
    if rbins.size ==1: rbins = np.array([rbins])

    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1]!=np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1*N2,))  # store radial pair separations
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i*N2:i*N2+N2] = distance(x1, x2, period)

    #sort results
    dd.sort()
    #count number less than r
    n = np.zeros((rbins.size,), dtype=np.int)
    for i in range(rbins.size):
        if rbins[i]>np.min(period)/2.0:
            print("r=", rbins[i], "  min(period)/2=", np.min(period)/2.0)
        n[i] = len(np.where(dd<=rbins[i])[0])

    return n


def xy_z_npairs(sample1, sample2, rp_bins, pi_bins, period=None):
    """
    Calculate the number of pairs with parellal separations less than or equal to
    pi_bins[i], and perpendicular separations less than or equal to rp_bins[i].

    Assumes the first N-1 dimensions are perpendicular to the line-of-sight (LOS), and
    the final dimension is parallel to the LOS.

    Parameters
    ----------
    sample1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    sample2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    rp_bins : array_like
        numpy array of boundaries defining the perpendicular bins in which pairs are
        counted.

    pi_bins : array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted.

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    Returns
    -------
    N_pairs : ndarray of shape (len(rp_bins),len(pi_bins))
        number counts of pairs

    """

    #work with arrays!
    sample1 = np.asarray(sample1)
    if sample1.ndim ==1: sample1 = np.array([sample1])
    sample2 = np.asarray(sample2)
    if sample2.ndim ==1: sample2 = np.array([sample2])
    rp_bins = np.asarray(rp_bins)
    if rp_bins.size ==1: rp_bins = np.array([rp_bins])
    pi_bins = np.asarray(pi_bins)
    if pi_bins.size ==1: pi_bins = np.array([pi_bins])

    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1]!=np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1*N2, 2))  # store pair separations
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i*N2:i*N2+N2, 1] = parallel_distance(x1, x2, period)
        dd[i*N2:i*N2+N2, 0] = perpendicular_distance(x1, x2, period)

    #count number less than r
    n = np.zeros((rp_bins.size, pi_bins.size), dtype=np.int)
    for i in range(rp_bins.size):
        for j in range(pi_bins.size):
            n[i, j] = np.sum((dd[:, 0]<=rp_bins[i]) & (dd[:, 1]<=pi_bins[j]))

    return n


def wnpairs(sample1, sample2, r, period=None, weights1=None, weights2=None):
    """
    Calculate the weighted number of pairs with separations less than or equal to rbins[i].

    Parameters
    ----------
    sample1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    sample2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted.
        len(rbins) = Nrbins + 1.

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    weights1 : array_like, optional
        length N1 array containing weights used for weighted pair counts, w1*w2.

    weights2 : array_like, optional
        length N2 array containing weights used for weighted pair counts, w1*w2.

    Returns
    -------
    wN_pairs : array of length len(rbins)
        weighted number counts of pairs

    """

    #work with arrays!
    sample1 = np.asarray(sample1)
    if sample1.ndim ==1: sample1 = np.array([sample1])
    sample2 = np.asarray(sample2)
    if sample2.ndim ==1: sample2 = np.array([sample2])
    r = np.asarray(r)
    if r.size == 1: r = np.array([r])

    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1]!=np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        if np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(sample1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(sample1)[0]:
            raise ValueError("weights1 should have same len as sample1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(sample2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(sample2)[0]:
            raise ValueError("weights2 should have same len as sample2")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1, N2), dtype=np.float64)  # store radial pair separations
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i, :] = distance(x1, x2, period)

    #count number less than r
    n = np.zeros((r.size,), dtype=np.float64)
    for i in range(r.size):
        if r[i]>np.min(period)/2:
            print("r=", r[i], "  min(period)/2=", np.min(period)/2)
        for j in range(N1):
            n[i] += np.sum(np.extract(dd[j, :]<=r[i], weights2))*weights1[j]

    return n


def xy_z_wnpairs(sample1, sample2, rp_bins, pi_bins, period=None, weights1=None, weights2=None):
    """
    Calculate the number of weighted pairs with parellal separations less than or equal to
    pi_bins[i], and perpendicular separations less than or equal to rp_bins[i].

    Assumes the first N-1 dimensions are perpendicular to the line-of-sight (LOS), and
    the final dimension is parallel to the LOS.

    Parameters
    ----------
    sample1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    sample2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    rp_bins : array_like
        numpy array of boundaries defining the perpendicular bins in which pairs are
        counted.

    pi_bins : array_like
        numpy array of boundaries defining the parallel bins in which pairs are counted.

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    weights1 : array_like, optional
        length N1 array containing weights used for weighted pair counts, w1*w2.

    weights2 : array_like, optional
        length N2 array containing weights used for weighted pair counts, w1*w2.


    Returns
    -------
    wN_pairs : ndarray of shape (len(rp_bins),len(pi_bins))
        weighted number counts of pairs

    """

    #work with arrays!
    sample1 = np.asarray(sample1)
    if sample1.ndim ==1: sample1 = np.array([sample1])
    sample2 = np.asarray(sample2)
    if sample2.ndim ==1: sample2 = np.array([sample2])
    rp_bins = np.asarray(rp_bins)
    if rp_bins.size ==1: rp_bins = np.array([rp_bins])
    pi_bins = np.asarray(pi_bins)
    if pi_bins.size ==1: pi_bins = np.array([pi_bins])

    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1]!=np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(sample1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(sample1)[0]:
            raise ValueError("weights1 should have same len as sample1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(sample2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(sample2)[0]:
            raise ValueError("weights2 should have same len as sample2")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1*N2, 2))  # store pair separations
    ww = np.zeros((N1*N2, 1))  # store pair separations
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i*N2:i*N2+N2, 1] = parallel_distance(x1, x2, period)
        dd[i*N2:i*N2+N2, 0] = perpendicular_distance(x1, x2, period)
        ww[i*N2:i*N2+N2] = weights1[i]*weights2

    #count number less than r
    n = np.zeros((rp_bins.size, pi_bins.size), dtype=np.float64)
    for i in range(rp_bins.size):
        for j in range(pi_bins.size):
                n[i, j] += np.sum(np.extract((dd[:, 0]<=rp_bins[i]) & (dd[:, 1]<=pi_bins[j]), ww))

    return n


def pairs(sample1, r, sample2=None, period=None):
    """
    Calculate the pairs with separations less than or equal to rbins[i].

    Parameters
    ----------
    sample1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    r : float
        radius for which pairs are counted.

    sample2 : array_like, optional
        N by k numpy array of k-dimensional positions. Should be between zero and
        period

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    Returns
    -------
    pairs : Set of pairs (i,j), with i < j

    """

    #work with arrays!
    sample1 = np.asarray(sample1)
    if sample2 is None:
        sample2 = np.asarray(sample1)
        self_match=False
    else:
        sample2 = np.asarray(sample2)
        self_match=True

    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1]!=np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        if np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1, N2))  # store radial pair separations
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i, :] = distance(x1, x2, period)

    pairs = np.argwhere(dd<=r)

    spairs = set()
    for i in range(len(pairs)):
        if self_match is False:
            if pairs[i, 0] != pairs[i, 1]:
                spairs.add((min(pairs[i]), max(pairs[i])))
        if self_match is True:
            spairs.add((min(pairs[i]), max(pairs[i])))

    return spairs


def distance(x1, x2, period=None):
    """
    Find the Euclidean distance between x1 & x2, accounting for box periodicity.

    Parameters
    ----------
    x1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period

    x2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period.

    period : array_like
        Size of the simulation box along each dimension. Defines periodic boundary
        conditioning.  Must be axis aligned.

    Returns
    -------
    distance : array

    """

    #process inputs
    x1 = np.asarray(x1)
    if x1.ndim ==1: x1 = np.array([x1])
    x2 = np.asarray(x2)
    if x2.ndim ==1: x2 = np.array([x2])
    if period is None:
        period = np.array([np.inf]*np.shape(x1)[-1])

    #check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else: k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    distance = np.sqrt(np.sum(m*m, axis=len(np.shape(m))-1))

    return distance


def parallel_distance(x1, x2, period=None):
    """
    Find the parallel distance between x1 & x2, accounting for box periodicity.

    Assumes the last dimension is the line-of-sight.

    Parameters
    ----------
    x1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period

    x2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period.

    period : array_like
        Size of the simulation box along each dimension. Defines periodic boundary
        conditioning.  Must be axis aligned.

    Returns
    -------
    distance : array

    """

    #process inputs
    x1 = np.asarray(x1)
    if x1.ndim ==1: x1 = np.array([x1])
    x2 = np.asarray(x2)
    if x2.ndim ==1: x2 = np.array([x2])
    if period is None:
        period = np.array([np.inf]*np.shape(x1)[-1])

    #check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else: k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(np.fabs(x1[:, -1] - x2[:, -1]), period[-1] - np.fabs(x1[:, -1] - x2[:, -1]))
    distance = np.sqrt(m*m)

    return distance


def perpendicular_distance(x1, x2, period=None):
    """
    Find the perpendicular distance between x1 & x2, accounting for box periodicity.

    Assumes the first N-1 dimensions are perpendicular to the line-of-sight.

    Parameters
    ----------
    x1 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period

    x2 : array_like
        N by k numpy array of k-dimensional positions. Should be between zero and period.

    period : array_like
        Size of the simulation box along each dimension. Defines periodic boundary
        conditioning.  Must be axis aligned.

    Returns
    -------
    distance : array

    """

    #process inputs
    x1 = np.asarray(x1)
    if x1.ndim ==1: x1 = np.array([x1])
    x2 = np.asarray(x2)
    if x2.ndim ==1: x2 = np.array([x2])
    if period is None:
        period = np.array([np.inf]*np.shape(x1)[-1])

    #check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else: k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(np.fabs(x1[:, :-1] - x2[:, :-1]), period[:-1] - np.fabs(x1[:, :-1] - x2[:, :-1]))
    distance = np.sqrt(np.sum(m*m, axis=len(np.shape(m))-1))

    return distance
