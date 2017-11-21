"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy.stats import gaussian_kde


__all__ = ('kde_cdf', )


def kde_cdf(data, x):
    """ Estimate the cumulative distribution function P(> x) defined by an input data sample.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    x : ndarray
        Numpy array of shape (num_pts, )

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Examples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> x = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf(data, x)
    """
    kde = gaussian_kde(data)
    return np.fromiter((kde.integrate_box_1d(-np.inf, high) for high in np.atleast_1d(x)), dtype='f4')


def kde_cdf_interpol(data, x, npts_interpol=None, npts_sample=None, weights_sample=None):
    """ Estimate the cumulative distribution function P(> x) defined by an input data sample,
    optionally interpolating from a downsampling of the data
    to improve performance at the cost of precision.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    x : ndarray
        Numpy array of shape (num_pts, )

    npts_interpol : int, optional
        Number of points used to build a lookup table in lieu of evaluating the CDF
        at every point in x. Should be smaller than the number of points in x.
        Default behavior is to evaluate the CDF exactly at each value of x.

    npts_sample : int, optional
        Size of downsampled data to use for construction of the KDE.
        Default behavior is not to downsample at all.
        Should be smaller than the number of points in the input data sample.

    weights_sample : ndarray, optional
        Numpy array of shape (num_sample, ) to use when downsampling the input data.
        This keyword is only operative when used in conjunction with the ``npts_sample`` argument.
        Default downsampling is random (without replacement).

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Examples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> x = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf_interpol(data, x)
    >>> cdf = kde_cdf_interpol(data, x, npts_interpol=500)
    >>> cdf = kde_cdf_interpol(data, x, npts_sample=1000)
    """
    if npts_sample is not None:
        msg = "npts_sample ={0} cannot exceed number of len(sample) = {1}"
        assert npts_sample <= len(data), msg.format(npts_sample, len(data))
        data = np.random.choice(data, npts_sample, replace=False, p=weights_sample)

    x = np.atleast_1d(x)
    if npts_interpol is not None:
        x_table = np.linspace(x.min(), x.max(), npts_interpol)
        cdf_table = kde_cdf(data, x_table)
        return np.interp(x, x_table, cdf_table)
    else:
        return kde_cdf(data, x)
