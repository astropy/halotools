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
