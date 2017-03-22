"""
This module contains the `TrivialProfile` class
used to assign the positions of central galaxies
to equal the positions of their host halos.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy import units as u
from astropy.constants import G

from ..profile_model_template import AnalyticDensityProf

from .... import model_defaults

from .....sim_manager import sim_defaults


newtonG = G.to(u.km*u.km*u.Mpc/(u.Msun*u.s*u.s))

__author__ = ['Andrew Hearin']

__all__ = ['TrivialProfile']


class TrivialProfile(AnalyticDensityProf):
    r""" Profile of dark matter halos with
    all their mass concentrated at exactly the halo center.

    """

    def __init__(self,
            cosmology=sim_defaults.default_cosmology,
            redshift=sim_defaults.default_redshift,
            mdef=model_defaults.halo_mass_definition,
            **kwargs):
        r"""
        Parameters
        ----------
        cosmology : object, optional
            Astropy cosmology object. Default is set in `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str, optional
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.

        Examples
        --------
        You can load a trivial profile model with the default settings simply by calling
        the class constructor with no arguments:

        >>> trivial_halo_prof_model = TrivialProfile()
        """

        super(TrivialProfile, self).__init__(cosmology, redshift, mdef)

    def dimensionless_mass_density(self, scaled_radius, total_mass):
        r"""
        Physical density of the halo scaled by the density threshold of the mass definition.

        The `dimensionless_mass_density` is defined as
        :math:`\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r}) / \rho_{\rm thresh}`,
        where :math:`\tilde{r}\equiv r/R_{\Delta}`.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        Returns
        -------
        dimensionless_density: array_like
            Dimensionless density of a dark matter halo
            at the input ``scaled_radius``, normalized by the
            `~halotools.empirical_models.profile_helpers.density_threshold`
            :math:`\rho_{\rm thresh}` for the
            halo mass definition, cosmology, and redshift.
            Result is an array of the dimension as the input ``scaled_radius``.

        """
        volume = (4*np.pi/3)*np.atleast_1d(scaled_radius)**3
        return total_mass/volume

    def enclosed_mass(self, radius, total_mass):
        r"""
        The mass enclosed within the input radius, :math:`M(<r) = 4\pi\int_{0}^{r}dr'r'^{2}\rho(r)`.

        For the `TrivialProfile`, this is equal to the total mass of the halo for all non-zero radii.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        Returns
        ----------
        enclosed_mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``radius``.
        """
        radius = np.atleast_1d(radius).astype(np.float64)
        total_mass = np.atleast_1d(total_mass).astype(np.float64)
        return np.where(radius > 0, total_mass, 0)
