r"""
Module containing the `~halotools.mock_observables.tpcf_multipole` function used to
calculate the multipoles of the redshift-space two-point correlation function,
aka the RSD multipoles.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import legendre

##########################################################################################

__all__ = ['tpcf_multipole']

__author__ = ['Duncan Campbell']


def tpcf_multipole(s_mu_tcpf_result, mu_bins, order=0):
    r"""
    Calculate the multipoles of the two point correlation function
    after first computing `~halotools.mock_observables.s_mu_tpcf`.

    Parameters
    ----------
    s_mu_tcpf_result : np.ndarray
        2-D array with the two point correlation function calculated in bins
        of :math:`s` and :math:`\mu`.  See `~halotools.mock_observables.s_mu_tpcf`.

    mu_bins : array_like
        array of :math:`\mu = \cos(\theta_{\rm LOS})`
        bins for which ``s_mu_tcpf_result`` has been calculated.
        Must be between [0,1].

    order : int, optional
        order of the multpole returned.

    Returns
    -------
    xi_l : np.array
        multipole of ``s_mu_tcpf_result`` of the indicated order.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube of length 250 Mpc/h.

    >>> Npts = 100
    >>> Lbox = 250.

    >>> x = np.random.uniform(0, Lbox, Npts)
    >>> y = np.random.uniform(0, Lbox, Npts)
    >>> z = np.random.uniform(0, Lbox, Npts)

    We transform our *x, y, z* points into the array shape used by the pair-counter by
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation
    is used throughout the `~halotools.mock_observables` sub-package:

    >>> sample1 = np.vstack((x,y,z)).T

    First, we calculate the correlation function using
    `~halotools.mock_observables.s_mu_tpcf`.

    >>> from halotools.mock_observables import s_mu_tpcf
    >>> s_bins  = np.linspace(0.01, 25, 10)
    >>> mu_bins = np.linspace(0, 1, 15)
    >>> xi_s_mu = s_mu_tpcf(sample1, s_bins, mu_bins, period=Lbox)

    Then, we can calculate the quadrapole of the correlation function:

    >>> xi_2 = tpcf_multipole(xi_s_mu, mu_bins, order=2)
    """

    # process inputs
    s_mu_tcpf_result = np.atleast_1d(s_mu_tcpf_result)
    mu_bins = np.atleast_1d(mu_bins)
    order = int(order)

    # calculate the center of each mu bin
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)

    # get the Legendre polynomial of the desired order.
    Ln = legendre(order)

    # numerically integrate over mu
    result = (2.0*order + 1.0)/2.0 * np.sum(s_mu_tcpf_result * np.diff(mu_bins) *\
        (Ln(mu_bin_centers) + Ln(-1.0*mu_bin_centers)), axis=1)

    return result
