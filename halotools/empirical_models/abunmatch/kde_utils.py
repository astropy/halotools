"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy.stats import gaussian_kde


__all__ = ('kde_cdf', )


def kde_cdf(data, y):
    """ Estimate the cumulative distribution function P(> y) defined by an input data sample.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    y : ndarray
        Numpy array of shape (num_pts, )

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Eyamples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> y = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf(data, y)
    """
    kde = gaussian_kde(data)
    return np.fromiter((kde.integrate_box_1d(-np.inf, high) for high in np.atleast_1d(y)), dtype='f4')


def kde_cdf_interpol(data, y, npts_interpol=None, npts_sample=None):
    """ Estimate the cumulative distribution function P(> y) defined by an input data sample,
    optionally interpolating from a downsampling of the data
    to improve performance at the cost of precision.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    y : ndarray
        Numpy array of shape (num_pts, )

    npts_interpol : int, optional
        Number of points used to build a lookup table in lieu of evaluating the CDF
        at every point in y. Should be smaller than the number of points in y.
        Default behavior is to evaluate the CDF eyactly at each value of y.

    npts_sample : int, optional
        Size of downsampled data to use for construction of the KDE.
        Default behavior is not to downsample at all.
        Should be smaller than the number of points in the input data sample.

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Eyamples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> y = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf_interpol(data, y)
    >>> cdf = kde_cdf_interpol(data, y, npts_interpol=500)
    >>> cdf = kde_cdf_interpol(data, y, npts_sample=1000)
    """
    if npts_sample is not None:
        msg = "npts_sample ={0} cannot eyceed number of len(sample) = {1}"
        assert npts_sample <= len(data), msg.format(npts_sample, len(data))
        data = np.random.choice(data, npts_sample, replace=False)

    y = np.atleast_1d(y)
    if npts_interpol is not None:
        y_table = np.linspace(y.min(), y.max(), npts_interpol)
        cdf_table = kde_cdf(data, y_table)
        return np.interp(y, y_table, cdf_table)
    else:
        return kde_cdf(data, y)


def kde_conditional_percentile(y, x, x_bins, npts_interpol=2000, npts_sample=5000):
    """ Estimate P(> y | x) for each input point (x, y) by using Gaussian kernel density
    estimation within each bin defined by x_bins.

    Points outside the bounds of x_bins
    will be treated as if they are members of the outer bins.

    Parameters
    ----------
    y : ndarray
        Numpy array of shape (num_sample, )

    x : ndarray
        Numpy array of shape (num_sample, )

    x_bins : ndarray
        Numpy array of shape (num_bins, )

    Examples
    --------
    >>> npts = int(1e4)
    >>> x = np.linspace(0, 10, npts)
    >>> y = np.random.normal(loc=x)
    >>> nbins = 15
    >>> x_bins = np.linspace(0, 10, nbins)
    >>> conditional_percentile = kde_conditional_percentile(y, x, x_bins)
    """
    result = np.zeros_like(y) + np.nan

    #  Overwrite values of x that lie outside the outermost bins
    x = np.where(x < x_bins[0], x_bins[0], x)
    dx = (x_bins[-1] - x_bins[-2])/2.
    x = np.where(x >= x_bins[-1], x_bins[-1]-dx, x)

    for low, high in zip(x_bins[:-1], x_bins[1:]):
        mask = (x >= low) & (x < high)
        npts_bin = np.count_nonzero(mask)
        if npts_bin > 0:
            data = y[mask]
            result[mask] = kde_cdf_interpol(data, data,
                npts_interpol=min(npts_interpol, npts_bin), npts_sample=min(npts_sample, npts_bin))

    return result
