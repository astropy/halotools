r"""
simple python brute force pair counting functions.  The primary purpose of these functions
is as a sanity check on more complex pair counting techniques.  These functions should not
be used on large data sets, as memory usage is very large, and runtimes can be very slow.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from ..pairs import distance
from halotools.utils import normalized_vectors

__all__ = ['cos2theta_pairs']
__author__ = ['Duncan Campbell']


def cos2theta_pairs(sample1, orientations1, sample2, rbins, period=None):
    r"""
    Calulate the sum of the square of the cosine of angles between vectors associetd with
    sample1 and the direction between sample1 and sample2.

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
    orientations1 = np.atleast_2d(orientations1)
    sample2 = np.atleast_2d(sample2)
    rbins = np.atleast_1d(rbins)

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error
    if np.shape(sample1)[-1] != np.shape(sample2)[-1]:
        raise ValueError("sample1 and sample2 inputs do not have the same dimension.")
        return None

    # Check to make sure both data sets have the same dimension. Otherwise, throw an error
    if np.shape(orientations1)[-1] != np.shape(sample1)[-1]:
        raise ValueError("orientations1 and sample1 inputs do not have the same dimension.")
        return None

    # Process period entry and check for consistency.
    if period is None:
        period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None

    # normalize orientation vectors
    orientations1 = normalized_vectors(orientations1)

    k = len(period)  # number of dimensions

    N1 = len(sample1)
    N2 = len(sample2)
    dd = np.zeros((N1*N2,))  # store radial pair separations
    oo = np.zeros((N1*N2, k))  # store orientation vectors for each pair
    vs = np.zeros((N1*N2, k))  # separation vectors for each pair
    for i in range(0, N1):  # calculate distance between every point and every other point
        x1 = sample1[i, :]
        x2 = sample2
        dd[i*N2:i*N2+N2] = distance(x1, x2, period)
        vs[i*N2:i*N2+N2, :] = r_12(x1, x2, period)
        oo[i*N2:i*N2+N2, :] = orientations1[i, :]

    # preform dot prouct between vs and orientations1
    vs = normalized_vectors(vs)  # normalize
    costheta = (vs*oo).sum(1)
    costheta[~np.isfinite(costheta)] = 0.0  # remove zero length vectors
    cos2theta = costheta*costheta
    print(costheta)

    # calculate sum of cos^2 theta in radial bins
    n = np.zeros((rbins.size,), dtype=np.int)
    omega = np.zeros((rbins.size,), dtype=np.float64)
    for i in range(rbins.size):
        inds = np.where(dd <= rbins[i])[0]
        n[i] = len(inds)
        omega[i] = np.sum(cos2theta[inds])

    return omega, n


def r_12(x1, x2, period=None):
    r"""
    Find the vector between x1 & x2, accounting for box periodicity.

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
    r_12 :  array
        array of vectors of x2 - x1
    """

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    if period is None:
        period = np.array([np.inf]*np.shape(x1)[-1])

    # check for consistency
    if np.shape(x1)[-1] != np.shape(x2)[-1]:
        raise ValueError("x1 and x2 list of points must have same dimension k.")
    else:
        k = np.shape(x1)[-1]
    if np.shape(period)[0] != np.shape(x1)[-1]:
        raise ValueError("period must have length equal to the dimension of x1 and x2.")

    m = x2 - x1
    for n in range(0, k):
        mask = (m[:, n] < -1.0*period[n]/2.0)
        m[:, n][mask] += period[n]
        mask = (m[:, n] > 1.0*period[n]/2.0)
        m[:, n][mask] -= period[n]

    return m



