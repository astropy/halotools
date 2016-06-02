"""
Module contains the `~halotools.empirical_models.IsotropicJeansVelocity` class
used to standardize the form of classes modeling satellite velocities.
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

from astropy.extern import six
from abc import ABCMeta, abstractmethod

__author__ = ['Andrew Hearin']

__all__ = ['IsotropicJeansVelocity']


@six.add_metaclass(ABCMeta)
class IsotropicJeansVelocity(object):
    """ Orthogonal mix-in class used to transform a configuration
    space model for the 1-halo term into a phase space model
    by solving the Jeans equation of the underlying potential.

    Notes
    ------
    This is intended to be a general purpose super-class providing a solution
    to the isotropic jeans equation for *any* spherically symmetric potential.
    Currently, this general-purpose functionality is not implemented and
    `IsotropicJeansVelocity` has no functionality of its own. The only
    analytical velocity model in Halotools is
    `~halotools.empirical_models.NFWJeansVelocity`,
    which over-rides the fundamental `dimensionless_radial_velocity_dispersion` method with
    an analytical solution to the Jeans equation for unbiased tracers orbiting in
    an equilibrated NFW potential.
    """

    def __init__(self, **kwargs):
        """
        """
        pass

    @abstractmethod
    def dimensionless_radial_velocity_dispersion(self, scaled_radius, *profile_params):
        """
        Method returns the radial velocity dispersion scaled by
        the virial velocity, as a function of the
        halo-centric distance scaled by the halo radius.

        Parameters
        ----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\\Delta}`, so that
            :math:`0 <= \\tilde{r} \\equiv r/R_{\\Delta} <= 1`. Can be a scalar or numpy array.

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s).
            In the simplest case, this sequence has a single element,
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies.
            More generally, there should be a ``profile_params`` sequence item for
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.prof_param_keys``.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.

        """
        pass

    def radial_velocity_dispersion(self, radius, total_mass, *profile_params):
        """
        Method returns the radial velocity dispersion scaled by
        the virial velocity as a function of the halo-centric distance.

        Parameters
        ----------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a number or a numpy array.

        *profile_params : Sequence of arrays
            Sequence of length-Ngals array(s) containing the input profile parameter(s).
            In the simplest case, this sequence has a single element,
            e.g. a single array storing values of the NFW concentrations of the Ngals galaxies.
            More generally, there should be a ``profile_params`` sequence item for
            every parameter in the profile model, each item a length-Ngals array.
            The sequence must have the same order as ``self.prof_param_keys``.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile as a function of the input ``radius``,
            in units of km/s.

        """
        virial_velocities = self.virial_velocity(total_mass)
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scaled_radius = radius/halo_radius

        dimensionless_velocities = (
            self.dimensionless_radial_velocity_dispersion(
                scaled_radius, *profile_params)
            )
        return dimensionless_velocities*virial_velocities
