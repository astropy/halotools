r"""
simple python brute force pair counting functions.  The primary purpose of these functions
is as a sanity check on more complex pair counting techniques.  These functions should not
be used on large data sets, as memory usage is very large, and runtimes can be very slow.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ["npairs", "wnpairs", "xy_z_npairs", "xy_z_wnpairs", "s_mu_npairs"]
__author__ = ["Duncan Campbell"]


def npairs(sample1, sample2, rbins, period=None):
    r"""
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

    sample1 = np.atleast_2d(sample1)
    sample2 = np.atleast_2d(sample2)
    rbins = np.atleast_1d(rbins)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf] * np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period] * np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1 * N2,))  # store radial pair separations
    for i in range(
        0, N1
    ):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i * N2 : i * N2 + N2] = distance(x1, x2, period)

    # sort results
    dd.sort()

    # count number less than r
    n = np.zeros((rbins.size,)).astype(int)
    for i in range(rbins.size):
        n[i] = len(np.where(dd <= rbins[i])[0])

    return n


def xy_z_npairs(sample1, sample2, rp_bins, pi_bins, period=None):
    r"""
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

    sample1 = np.atleast_2d(sample1)
    sample2 = np.atleast_2d(sample2)
    rp_bins = np.atleast_1d(rp_bins)
    pi_bins = np.atleast_1d(pi_bins)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf] * np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period] * np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1 * N2, 2))  # store pair separations
    for i in range(
        0, N1
    ):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i * N2 : i * N2 + N2, 1] = parallel_distance(x1, x2, period)
        dd[i * N2 : i * N2 + N2, 0] = perpendicular_distance(x1, x2, period)

    # count number less than r
    n = np.zeros((rp_bins.size, pi_bins.size)).astype(int)
    for i in range(rp_bins.size):
        for j in range(pi_bins.size):
            n[i, j] = np.sum((dd[:, 0] <= rp_bins[i]) & (dd[:, 1] <= pi_bins[j]))

    return n


def wnpairs(sample1, sample2, r, period=None, weights1=None, weights2=None):
    r"""
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

    sample1 = np.atleast_2d(sample1)
    sample2 = np.atleast_2d(sample2)
    r = np.atleast_1d(r)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf] * np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period] * np.shape(sample1)[-1])
        if np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    # Process weights1 entry and check for consistency.
    if weights1 is None:
        weights1 = np.array([1.0] * np.shape(sample1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(sample1)[0]:
            raise ValueError("weights1 should have same len as sample1")
            return None
    # Process weights2 entry and check for consistency.
    if weights2 is None:
        weights2 = np.array([1.0] * np.shape(sample2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(sample2)[0]:
            raise ValueError("weights2 should have same len as sample2")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1, N2), dtype=np.float64)  # store radial pair separations
    for i in range(
        0, N1
    ):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i, :] = distance(x1, x2, period)

    # count number less than r
    n = np.zeros((r.size,), dtype=np.float64)
    for i in range(r.size):
        for j in range(N1):
            n[i] += np.sum(np.extract(dd[j, :] <= r[i], weights2)) * weights1[j]

    return n


def xy_z_wnpairs(
    sample1, sample2, rp_bins, pi_bins, period=None, weights1=None, weights2=None
):
    r"""
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

    sample1 = np.atleast_2d(sample1)
    sample2 = np.atleast_2d(sample2)
    rp_bins = np.atleast_1d(rp_bins)
    pi_bins = np.atleast_1d(pi_bins)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf] * np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period] * np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    # Process weights1 entry and check for consistency.
    if weights1 is None:
        weights1 = np.array([1.0] * np.shape(sample1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(sample1)[0]:
            raise ValueError("weights1 should have same len as sample1")
            return None
    # Process weights2 entry and check for consistency.
    if weights2 is None:
        weights2 = np.array([1.0] * np.shape(sample2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(sample2)[0]:
            raise ValueError("weights2 should have same len as sample2")
            return None

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1 * N2, 2))  # store pair separations
    ww = np.zeros((N1 * N2, 1))  # store pair separations
    for i in range(
        0, N1
    ):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i * N2 : i * N2 + N2, 1] = parallel_distance(x1, x2, period)
        dd[i * N2 : i * N2 + N2, 0] = perpendicular_distance(x1, x2, period)
        ww[i * N2 : i * N2 + N2] = weights1[i] * weights2

    # count number less than r
    n = np.zeros((rp_bins.size, pi_bins.size), dtype=np.float64)
    for i in range(rp_bins.size):
        for j in range(pi_bins.size):
            n[i, j] += np.sum(
                np.extract((dd[:, 0] <= rp_bins[i]) & (dd[:, 1] <= pi_bins[j]), ww)
            )

    return n


def s_mu_npairs(sample1, sample2, s_bins, mu_bins, period=None):
    r"""
    Calculate the number of pairs with 3D radial separations less than or equal to
    :math:`s`, and angular separations along the LOS, :math:`\mu=\cos(\theta_{\rm LOS})`.

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

    s_bins : array_like
        numpy array of shape (num_s_bin_edges, ) storing the :math:`s`
        boundaries defining the bins in which pairs are counted.

    mu_bins : array_like
        numpy array of shape (num_mu_bin_edges, ) storing the
        :math:`\cos(\theta_{\rm LOS})` boundaries defining the bins in
        which pairs are counted. All values must be between [0,1].

    period : array_like, optional
        length k array defining  periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.

    Returns
    -------
    N_pairs : ndarray of shape (num_s_bin_edges, num_mu_bin_edges) storing the
        number counts of pairs with separations less than ``s_bins`` and ``mu_bins``

    Notes
    -----
    Along the first dimension of ``N_pairs``, :math:`s` (the radial separation) increases.
    Along the second dimension,  :math:`\mu` (the cosine of :math:`\theta_{\rm LOS}`)
    decreases, i.e. :math:`\theta_{\rm LOS}` increases.
    """

    sample1 = np.atleast_2d(sample1)
    sample2 = np.atleast_2d(sample2)
    s_bins = np.atleast_1d(s_bins)
    mu_bins = np.atleast_1d(mu_bins)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf] * np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period] * np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    # create N1 x N2 x 2 array to store **all** pair separation distances
    # note that this array can be very large for large N1 and N2
    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1 * N2, 2))

    # calculate distance between every point and every other point
    for i in range(0, N1):
        x1 = sample1[i, :]
        x2 = sample2
        dd[i * N2 : i * N2 + N2, 0] = distance(x1, x2, period)
        dd[i * N2 : i * N2 + N2, 1] = np.cos(theta_LOS(x1, x2, period))

    # put mu bins in increasing theta_LOS order
    mu_bins = np.sort(mu_bins)[::-1]

    # bin distances in s and mu bins
    n = np.zeros((s_bins.size, mu_bins.size)).astype(int)
    for i in range(s_bins.size):
        for j in range(mu_bins.size):
            n[i, j] = np.sum((dd[:, 0] <= s_bins[i]) & (dd[:, 1] >= mu_bins[j]))

    return n


def distance(x1, x2, period=None):
    r"""
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

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    if period is None:
        period = np.array([np.inf] * np.shape(x1)[-1])

    # check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else:
        k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    distance = np.sqrt(np.sum(m * m, axis=len(np.shape(m)) - 1))

    return distance


def parallel_distance(x1, x2, period=None):
    r"""
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

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    if period is None:
        period = np.array([np.inf] * np.shape(x1)[-1])

    # check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else:
        k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(
        np.fabs(x1[:, -1] - x2[:, -1]), period[-1] - np.fabs(x1[:, -1] - x2[:, -1])
    )
    distance = np.sqrt(m * m)

    return distance


def perpendicular_distance(x1, x2, period=None):
    r"""
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

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    if period is None:
        period = np.array([np.inf] * np.shape(x1)[-1])

    # check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else:
        k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = np.minimum(
        np.fabs(x1[:, :-1] - x2[:, :-1]), period[:-1] - np.fabs(x1[:, :-1] - x2[:, :-1])
    )
    distance = np.sqrt(np.sum(m * m, axis=len(np.shape(m)) - 1))

    return distance


def theta_LOS(x1, x2, period=None):
    r"""
    Find the separation angle from the LOS between x1 & x2, accounting for box periodicity.

    Assumes the first N-1 dimensions are perpendicular to the line-of-sight (LOS).

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
    theta_LOS : array
        angle from LOS in radians

    Notes
    -----
    theta_LOS is set to 0.0 if the distance between points is 0.0
    """

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    if period is None:
        period = np.array([np.inf] * np.shape(x1)[-1])

    # check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else:
        k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    r_perp = perpendicular_distance(x1, x2, period=period)
    r_parallel = parallel_distance(x1, x2, period=period)

    # deal with zero separation
    r = np.sqrt(r_perp**2 + r_parallel**2)
    mask = r > 0.0

    theta = np.zeros(len(r))  # set to zero if r==0
    theta[mask] = np.pi / 2.0 - np.arctan2(r_parallel[mask], r_perp[mask])

    return theta
