""" Module containing independently written code to calculate galaxy-galaxy lensing,
used in the unit-testing of `~halotools.mock_observables.surface_density` sub-package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from ....sim_manager.sim_defaults import default_cosmology


__all__ = ('external_delta_sigma', )
__author__ = ('Surhud More', )


def external_delta_sigma(galaxies, particles, rp_bins, period, projection_period,
        cosmology=default_cosmology):
    r"""
    Parameters
    ----------
    galaxies : array_like
        Ngal x 2 numpy array containing 2-d positions of galaxies.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    particles : array_like
        Npart x 2 numpy array containing 2-d positions of particles.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools. Assumes constant particle masses, but can
        use weighted pair counts as scipy 0.19.0 is released.
        scipy.spatial.cKDTree will acquire a weighted pair count functionality

    rp_bins : array_like
        array of projected radial boundaries defining the bins in which the result is
        calculated.  The minimum of rp_bins must be > 0.0.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    period : array_like
        Length-2 sequence defining the periodic boundary conditions
        in each dimension. If you instead provide a single scalar, Lbox,
        period is assumed to be the same in all Cartesian directions.
        Length units are comoving and assumed to be in Mpc/h,
        here and throughout Halotools.

    projection_period : float
        The period along the direction of projection

    cosmology : instance of `astropy.cosmology`, optional
        Default value is set in `~halotools.sim_manager.default_cosmology` module.
        Typically you should use the `cosmology` attribute of the halo catalog
        you used to populate mock galaxies.

    Returns
    -------
    rmids : np.array
        The bins at which :math:`\Delta\Sigma` is calculated.
        The units of `rmids` is :math:`hinv Mpc`, where distances are in comoving units.
        You can convert to physical units using the input cosmology and redshift.
        Note that little h = 1 here and throughout Halotools.

    Delta_Sigma : np.array
        :math:`\Delta\Sigma(r_p)` calculated at projected comoving radial distances ``rp_bins``.
        The units of `ds` are :math:`h * M_{\odot} / Mpc^2`, where distances are in comoving units.
        You can convert to physical units using the input cosmology and redshift.
        Note that little h = 1 here and throughout Halotools.

    Notes
    -----
    :math:`\Delta\Sigma` is calculated by first calculating the projected
    surface density :math:`\Sigma` using the particles passed to the code

    and then,

    .. math::

        \Delta\Sigma(r_p) = \bar{\Sigma}(<r_p) - \Sigma(r_p)
    """

    from scipy.spatial import cKDTree
    from astropy.constants import G

    Ngal = float(galaxies.shape[0])
    Npart = float(particles.shape[0])
    if np.isscalar(period):
        Area = period**2
    else:
        Area = period[0] * period[1]

    tree = cKDTree(galaxies, boxsize=period)
    ptree = cKDTree(particles, boxsize=period)
    pairs_inside_rad = tree.count_neighbors(ptree, rp_bins)

    pairs_in_annuli = np.diff(pairs_inside_rad)

    # rhobar = 3H0^2/(8 pi G) Om0
    rhobar = 3.e4/(8*np.pi*G.to('km^2 Mpc/(s^2 Msun)').value)*cosmology.Om0

    sigmabar = rhobar*projection_period

    # This initializes sigma(rmids)
    rmids = rp_bins[1:]/2+rp_bins[:-1]/2
    xi2d = pairs_in_annuli/(Ngal*Npart/Area*(np.pi*(rp_bins[1:]**2-rp_bins[:-1]**2))) - 1.0
    sigma = sigmabar*xi2d

    # Now initialize sigmainside(rp_bins)
    xi2dinside = pairs_inside_rad/(Npart*Ngal/Area*(np.pi*rp_bins**2)) - 1.0
    sigmainside = sigmabar*xi2dinside

    from scipy.interpolate import interp1d
    spl = interp1d(np.log(rp_bins), np.log(sigmainside), kind="cubic")

    return rmids, np.exp(spl(np.log(rmids)))-sigma
