"""
This module contains the `AnalyticalDensityProf` class,
a container class for the distribution of mass and/or galaxies
within dark matter halos.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy.integrate import quad as quad_integration
from scipy.optimize import minimize as scipy_minimize
from astropy import units as u
from astropy.constants import G

from . import halo_boundary_functions

from ... import model_defaults

newtonG = G.to(u.km * u.km * u.Mpc / (u.Msun * u.s * u.s))

__author__ = ["Andrew Hearin", "Benedikt Diemer"]

__all__ = ["AnalyticDensityProf"]


class AnalyticDensityProf(object):
    r"""Container class for any analytical radial profile model.

    See :ref:`profile_template_tutorial` for a review of the mathematics of
    halo profiles, and a thorough description of how the relevant equations
    are implemented in the `AnalyticDensityProf` source code.

    Notes
    -----
    The primary behavior of the `AnalyticDensityProf` class is governed by the
    `dimensionless_mass_density`  method. The `AnalyticDensityProf` class has no
    implementation of its own of `dimensionless_mass_density`, but does implement
    all other behaviors that derive from `dimensionless_mass_density`. Thus for users
    who wish to define their own profile class, defining the `dimensionless_mass_density` of
    the profile is the necessary and sufficient ingredient.
    """

    def __init__(self, cosmology, redshift, mdef, halo_boundary_key=None, **kwargs):
        r"""
        Parameters
        -----------
        cosmology : object
            Instance of an `~astropy.cosmology` object.

        redshift: array_like
            Can be a scalar or a numpy array.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'.

        halo_boundary_key : str, optional
            Default behavior is to use the column associated with the input mdef.

        """
        self.cosmology = cosmology
        self.redshift = redshift
        self.mdef = mdef

        # The following four attributes are derived quantities from the above,
        # so that self-consistency between them is ensured
        self.density_threshold = halo_boundary_functions.density_threshold(
            cosmology=self.cosmology, redshift=self.redshift, mdef=self.mdef
        )
        if halo_boundary_key is None:
            self.halo_boundary_key = model_defaults.get_halo_boundary_key(self.mdef)
        else:
            self.halo_boundary_key = halo_boundary_key
        self.prim_haloprop_key = model_defaults.get_halo_mass_key(self.mdef)

        self.gal_prof_param_keys = []
        self.halo_prof_param_keys = []
        self.publications = []
        self.param_dict = {}

    def dimensionless_mass_density(self, scaled_radius, *prof_params):
        r"""
        Physical density of the halo scaled by the density threshold of the mass definition:

        The `dimensionless_mass_density` is defined as
        :math:`\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r}) / \rho_{\rm thresh}`,
        where :math:`\tilde{r}\equiv r/R_{\Delta}`.
        The quantity :math:`\rho_{\rm thresh}` is a function of
        the halo mass definition, cosmology and redshift,
        and is computed via the
        `~halotools.empirical_models.halo_boundary_functions.density_threshold` function.
        The quantity :math:`\rho_{\rm prof}` is the physical mass density of the
        halo profile and is computed via the `mass_density` function.

        See :ref:`halo_profile_definitions` for derivations and implementation details.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        *prof_params : array_like, optional
            Any additional array or sequence of arrays
            necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        -------
        dimensionless_density: array_like
            Dimensionless density of a dark matter halo
            at the input ``scaled_radius``, normalized by the
            `~halotools.empirical_models.halo_boundary_functions.density_threshold`
            :math:`\rho_{\rm thresh}` for the
            halo mass definition, cosmology, and redshift.
            Result is an array of the dimension as the input ``scaled_radius``.

        Notes
        -----
        All of the behavior of a subclass of `AnalyticDensityProf` is determined by
        `dimensionless_mass_density`. This is numerically convenient, because mass densities
        in physical units are astronomically large numbers, whereas `dimensionless_mass_density`
        is of order :math:`\mathcal{O}(1-100)`. This also saves users writing their own subclass
        from having to worry over factors of little h, how profile normalization scales
        with the mass definition, etc. Once a model's `dimensionless_mass_density` is specified,
        all the other functionality is derived from this definition.

        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        pass

    def mass_density(self, radius, mass, *prof_params):
        r"""
        Physical density of the halo at the input radius,
        given in units of :math:`h^{3}/{\rm Mpc}^{3}`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        -------
        density: array_like
            Physical density of a dark matter halo of the input ``mass``
            at the input ``radius``. Result is an array of the
            dimension as the input ``radius``, reported in units of :math:`h^{3}/Mpc^{3}`.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        halo_radius = self.halo_mass_to_halo_radius(mass)
        scaled_radius = radius / halo_radius

        dimensionless_mass = self.dimensionless_mass_density(
            scaled_radius, *prof_params
        )

        density = self.density_threshold * dimensionless_mass
        return density

    def _enclosed_dimensionless_mass_integrand(self, scaled_radius, *prof_params):
        r"""
        Integrand used when computing `cumulative_mass_PDF`.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        -------
        integrand: array_like
            function to be integrated to yield the amount of enclosed mass.
        """
        dimensionless_density = self.dimensionless_mass_density(
            scaled_radius, *prof_params
        )
        return dimensionless_density * 4 * np.pi * scaled_radius**2

    def cumulative_mass_PDF(self, scaled_radius, *prof_params):
        r"""
        The fraction of the total mass enclosed within dimensionless radius,

        :math:`P_{\rm prof}(<\tilde{r}) \equiv M_{\Delta}(<\tilde{r}) / M_{\Delta},`
        where :math:`\tilde{r} \equiv r / R_{\Delta}`.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed
            within radius x, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``x``.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        x = np.atleast_1d(scaled_radius).astype(np.float64)
        enclosed_mass = np.zeros_like(x)

        for i in range(len(x)):
            enclosed_mass[i], _ = quad_integration(
                self._enclosed_dimensionless_mass_integrand,
                0.0,
                x[i],
                epsrel=1e-5,
                args=prof_params,
            )

        total, _ = quad_integration(
            self._enclosed_dimensionless_mass_integrand,
            0.0,
            1.0,
            epsrel=1e-5,
            args=prof_params,
        )

        return enclosed_mass / total

    def enclosed_mass(self, radius, total_mass, *prof_params):
        r"""
        The mass enclosed within the input radius.

        :math:`M(<r) = 4\pi\int_{0}^{r}dr'r'^{2}\rho(r)`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        ----------
        enclosed_mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``radius``.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        radius = np.atleast_1d(radius).astype(np.float64)
        scaled_radius = radius / self.halo_mass_to_halo_radius(total_mass)
        mass = self.cumulative_mass_PDF(scaled_radius, *prof_params) * total_mass

        return mass

    def dimensionless_circular_velocity(self, scaled_radius, *prof_params):
        r"""Circular velocity scaled by the virial velocity,
        :math:`V_{\rm cir}(x) / V_{\rm vir}`, as a function of
        dimensionless position :math:`\tilde{r} = r / R_{\rm vir}`.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        -------
        vcir : array_like
            Circular velocity scaled by the virial velocity,
            :math:`V_{\rm cir}(x) / V_{\rm vir}`.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        return np.sqrt(
            self.cumulative_mass_PDF(scaled_radius, *prof_params) / scaled_radius
        )

    def virial_velocity(self, total_mass):
        r"""The circular velocity evaluated at the halo boundary,
        :math:`V_{\rm vir} \equiv \sqrt{GM_{\rm halo}/R_{\rm halo}}`.

        Parameters
        --------------
        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array.

        Returns
        --------
        vvir : array_like
            Virial velocity in km/s.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        return halo_boundary_functions.halo_mass_to_virial_velocity(
            total_mass, self.cosmology, self.redshift, self.mdef
        )

    def circular_velocity(self, radius, total_mass, *prof_params):
        r"""
        The circular velocity, :math:`V_{\rm cir} \equiv \sqrt{GM(<r)/r}`,
        as a function of halo-centric distance r.

        Parameters
        --------------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        *prof_params : array_like, optional
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        ----------
        vc: array_like
            The circular velocity in km/s; has the same dimensions as the input ``radius``.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scaled_radius = np.atleast_1d(radius) / halo_radius
        return self.dimensionless_circular_velocity(
            scaled_radius, *prof_params
        ) * self.virial_velocity(total_mass)

    def _vmax_helper(self, scaled_radius, *prof_params):
        """Helper function used to calculate `vmax` and `rmax`."""
        encl = self.cumulative_mass_PDF(scaled_radius, *prof_params)
        return -1.0 * encl / scaled_radius

    def rmax(self, total_mass, *prof_params):
        r"""Radius at which the halo attains its maximum circular velocity.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        *prof_params : array_like
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        --------
        rmax : array_like
            :math:`R_{\rm max}` in Mpc/h.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        guess = 0.25

        result = scipy_minimize(self._vmax_helper, guess, args=prof_params)

        return result.x[0] * halo_radius

    def vmax(self, total_mass, *prof_params):
        r"""Maximum circular velocity of the halo profile.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        *prof_params : array_like
            Any additional array(s) necessary to specify the shape of the radial profile,
            e.g., halo concentration.

        Returns
        --------
        vmax : array_like
            :math:`V_{\rm max}` in km/s.

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        guess = 0.25
        result = scipy_minimize(self._vmax_helper, guess, args=prof_params)
        halo_radius = self.halo_mass_to_halo_radius(total_mass)

        return self.circular_velocity(
            result.x[0] * halo_radius, total_mass, *prof_params
        )

    def halo_mass_to_halo_radius(self, total_mass):
        r"""
        Spherical overdensity radius as a function of the input mass.

        Note that this function is independent of the form of the density profile.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        Returns
        -------
        radius : array_like
            Radius of the halo in Mpc/h units.
            Will have the same dimension as the input ``total_mass``.

        Notes
        ------
        The behavior of this function derives from
        `~halotools.empirical_models.halo_mass_to_halo_radius`.

        """
        return halo_boundary_functions.halo_mass_to_halo_radius(
            total_mass, cosmology=self.cosmology, redshift=self.redshift, mdef=self.mdef
        )

    def halo_radius_to_halo_mass(self, radius):
        r"""
        Spherical overdensity mass as a function of the input radius.

        Note that this function is independent of the form of the density profile.

        Parameters
        ------------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a number or a numpy array.

        Returns
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`.
            Will have the same dimension as the input ``radius``.

        Notes
        ------
        The behavior of this function derives from
        `~halotools.empirical_models.halo_radius_to_halo_mass`.

        """
        return halo_boundary_functions.halo_radius_to_halo_mass(
            radius, cosmology=self.cosmology, redshift=self.redshift, mdef=self.mdef
        )
