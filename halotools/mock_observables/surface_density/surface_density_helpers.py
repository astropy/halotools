""" Helper functions used in the surface_density sub-package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy import units as u

__all__ = ('annular_area_weighted_midpoints', 'log_interpolation_with_inner_zero_masking',
        'rho_matter_comoving_in_halotools_units')
__author__ = ('Andrew Hearin', 'Shun Saito')


def annular_area_weighted_midpoints(rp_bins):
    r""" Calculate the radius that bisects the areas a set of circles.

    Parameters
    ----------
    rp_bins : array_like
        Array of shape (num_circles, ) defining the input circles

    Returns
    -------
    rp_mids : array_like
        Array of shape (num_circles-1, ) defining the radii of annuli
        that subtend equal areas.

    Examples
    ---------
    >>> r_mids = annular_area_weighted_midpoints(np.linspace(0.1, 5, 10))
    """
    return np.sqrt(0.5*(rp_bins[:-1]**2 + rp_bins[1:]**2))


def log_interpolation_with_inner_zero_masking(onep_sig_in, rp_bins, rp_mids):
    r""" Given an array ``onep_sig_in`` whose values are tabulated at
    ``rp_bins``, interolate in log-space to evaluate the array values at
    ``rp_mids``, taking care to mask over any zeros in ``onep_sig_in``.

    Parameters
    ----------
    onep_sig_in : array
        Ndarray of shape (num_rp_bins, )

    rp_bins : array
        Ndarray of shape (num_rp_bins, )

    rp_mids : array
        Ndarray of shape (num_rp_bins-1, )

    Returns
    --------
    result : array
        Ndarray of shape (num_rp_bins-1, )
    """

    first_nonzero_idx = len(onep_sig_in) - np.count_nonzero(onep_sig_in)
    x2 = np.log(rp_mids[first_nonzero_idx:])
    xp2 = np.log(rp_bins[first_nonzero_idx:])
    fp2 = np.log(onep_sig_in[first_nonzero_idx:])
    onep_sig_out = np.zeros_like(rp_mids)
    onep_sig_out[first_nonzero_idx:] = np.exp(np.interp(x2, xp2, fp2))[:]
    return onep_sig_out


def rho_matter_comoving_in_halotools_units(cosmology):
    r""" Calculate the comoving matter density in units of
    :math:`M_{\odot}/{\rm Mpc}^3` assuming :math:`h = 1`.

    Parameters
    ----------
    cosmology : object
        Astropy `~astropy.cosmology` object

    Returns
    -------
    mean_rho_comoving : float

    Examples
    ---------
    >>> from astropy.cosmology import Planck15
    >>> mean_rho_comoving = rho_matter_comoving_in_halotools_units(Planck15)
    """
    rho_crit0 = cosmology.critical_density0
    rho_crit0 = rho_crit0.to(u.Msun/u.Mpc**3).value/cosmology.h**2
    mean_rho_comoving = (cosmology.Om0 + cosmology.Onu0)*rho_crit0
    return mean_rho_comoving
