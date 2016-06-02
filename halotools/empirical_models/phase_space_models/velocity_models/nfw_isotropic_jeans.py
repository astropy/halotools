"""
Module contains the `~halotools.empirical_models.NFWJeansVelocity` class
used to model the velocities of satellite galaxies orbiting in Jeans equlibrium
in an NFW potential.
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.integrate import quad as quad_integration

from .isotropic_jeans_model_template import IsotropicJeansVelocity

from ....utils.array_utils import convert_to_ndarray

__author__ = ['Andrew Hearin']

__all__ = ['NFWJeansVelocity']


class NFWJeansVelocity(IsotropicJeansVelocity):
    """ Orthogonal mix-in class providing the solution to the Jeans equation
    for galaxies orbiting in an isotropic NFW profile with no spatial bias.
    """

    def __init__(self, **kwargs):
        """
        """
        IsotropicJeansVelocity.__init__(self, **kwargs)

    def _jeans_integrand_term1(self, y):
        """
        """
        return np.log(1+y)/(y**3*(1+y)**2)

    def _jeans_integrand_term2(self, y):
        """
        """
        return 1/(y**2*(1+y)**3)

    def dimensionless_radial_velocity_dispersion(self, scaled_radius, *conc):
        """
        Analytical solution to the isotropic jeans equation for an NFW potential,
        rendered dimensionless via scaling by the virial velocity.

        :math:`\\tilde{\\sigma}^{2}_{r}(\\tilde{r})\\equiv\\sigma^{2}_{r}(\\tilde{r})/V_{\\rm vir}^{2} = \\frac{c^{2}\\tilde{r}(1 + c\\tilde{r})^{2}}{g(c)}\int_{c\\tilde{r}}^{\infty}{\\rm d}y\\frac{g(y)}{y^{3}(1 + y)^{2}}`

        See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

        Parameters
        -----------
        scaled_radius : array_like
            Length-Ngals numpy array storing the halo-centric distance
            *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`.

        total_mass: array_like
            Length-Ngals numpy array storing the halo mass in :math:`M_{\odot}/h`.

        conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.
        """
        x = convert_to_ndarray(scaled_radius, dt=np.float64)
        result = np.zeros_like(x)

        prefactor = conc*(conc*x)*(1. + conc*x)**2/self.g(conc)

        lower_limit = conc*x
        upper_limit = float("inf")
        for i in range(len(x)):
            term1, _ = quad_integration(self._jeans_integrand_term1,
                lower_limit[i], upper_limit, epsrel=1e-5)
            term2, _ = quad_integration(self._jeans_integrand_term2,
                lower_limit[i], upper_limit, epsrel=1e-5)
            result[i] = term1 - term2

        return np.sqrt(result*prefactor)

    def radial_velocity_dispersion(self, radius, total_mass, conc):
        """
        Method returns the radial velocity dispersion scaled by
        the virial velocity as a function of the halo-centric distance.

        Parameters
        ----------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a number or a numpy array.

        conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile as a function of the input ``radius``,
            in units of km/s.

        """
        return IsotropicJeansVelocity.radial_velocity_dispersion(
            self, radius, total_mass, conc)
