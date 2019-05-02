"""
Module containing classes used to model the mapping between
stellar mass and halo mass based on Behroozi et al. (2010).
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from warnings import warn

from .smhm_helpers import safely_retrieve_redshift

from .. import model_helpers as model_helpers
from ..component_model_templates import PrimGalpropModel

__all__ = ['Behroozi10SmHm']


class Behroozi10SmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on
    `Behroozi et al 2010 <http://arxiv.org/abs/astro-ph/1001.0015/>`_.

    .. note::

        The `Behroozi10SmHm` model is part of the ``behroozi10``
        prebuilt composite subhalo-based model. For a tutorial on the ``behroozi10``
        composite model, see :ref:`behroozi10_composite_model`.

    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        scatter_model : object, optional
            Class governing stochasticity of stellar mass. Default scatter is log-normal,
            implemented by the `~halotools.empirical_models.LogNormalScatterModel` class.

        scatter_abscissa : array_like, optional
            Array of values giving the abscissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        scatter_ordinates : array_like, optional
            Array of values defining the level of scatter at the input abscissa.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation. Recommended default behavior
            is to leave this argument unspecified.

            If no ``redshift`` argument is given to the constructor, you will be free to use the
            analytical relations bound to `Behroozi10SmHm` to study the redshift-dependence
            of the SMHM by passing in a ``redshift`` argument to the `mean_log_halo_mass`
            and `mean_stellar_mass` methods.

            If you do pass a ``redshift`` argument to the constructor, the instance of the
            `Behroozi10SmHm` will only return results for this redshift, and will raise an
            exception if you attempt to pass in a redshift to these methods.
            See the Notes below to understand the motivation for this behavior.

        Notes
        ------
        Note that the `Behroozi10SmHm` class is a distinct from the `Behroozi10` model
        in several respects. The most important distinction to understand is that
        `Behroozi10` is a composite model that has been built to populate simulations
        with mock galaxies, whereas `Behroozi10SmHm` is a component model
        that is just a collection of analytical functions.

        Related to the above, the `Behroozi10` composite model has a single redshift
        hard-wired into its behavior to guarantee consistency with the
        halo catalog into which `Behroozi10` will sprinkle mock galaxies. On the other hand,
        the `Behroozi10SmHm` model need not have a ``redshift`` attribute bound to it at all, which permits
        you to use the analytical functions bound to `Behroozi10SmHm` to study the redshift-dependence
        of the stellar-to-halo-mass relation. However, since the `Behroozi10` composite model
        uses an instance of the `Behroozi10SmHm` for its stellar-to-halo-mass feature, then there must be
        some mechanism by which the redshift-dependence of the `Behroozi10SmHm` can be held fixed.
        The option to provide a specific redshift to the constructor of `Behroozi10SmHm`
        provides this mechanism.

        """

        self.littleh = 0.7

        super(Behroozi10SmHm, self).__init__(
            galprop_name='stellar_mass', **kwargs)

        self._methods_to_inherit.extend(['mean_log_halo_mass'])

        self.publications = ['arXiv:1001.0015']

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters
        set to the column 2 values in Table 2 of Behroozi et al. (2010).

        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        # All calculations are done internally using the same h=0.7 units
        # as in Behroozi et al. (2010), so the parameter values here are
        # the same as in Table 2, even though the mean_log_halo_mass and
        # mean_stellar_mass methods use accept and return arguments in h=1 units.

        d = ({'smhm_m0_0': 10.72,
            'smhm_m0_a': 0.59,
            'smhm_m1_0': 12.35,
            'smhm_m1_a': 0.3,
            'smhm_beta_0': 0.43,
            'smhm_beta_a': 0.18,
            'smhm_delta_0': 0.56,
            'smhm_delta_a': 0.18,
            'smhm_gamma_0': 1.54,
            'smhm_gamma_a': 2.52})

        return d

    def mean_log_halo_mass(self, log_stellar_mass, **kwargs):
        """ Return the halo mass of a central galaxy as a function
        of the stellar mass.

        Parameters
        ----------
        log_stellar_mass : array
            Array of base-10 logarithm of stellar masses in h=1 solar mass units.

        redshift : float or array, optional
            Redshift of the halo hosting the galaxy. If passing an array,
            must be of the same length as the input ``log_stellar_mass``.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Returns
        -------
        log_halo_mass : array_like
            Array containing 10-base logarithm of halo mass in h=1 solar mass units.

        Notes
        ------
        The parameter values in Behroozi+10 were fit to data assuming h=0.7,
        but all halotools inputs are in h=1 units. Thus we will transform our
        input stellar mass to h=0.7 units, evaluate using the behroozi parameters,
        and then transform back to h=1 units before returning the result.
        """
        redshift = safely_retrieve_redshift(self, 'mean_log_halo_mass', **kwargs)

        # convert mass from h=1 to h=0.7
        stellar_mass = (10.**log_stellar_mass)/(self.littleh**2)
        a = 1./(1. + redshift)

        logm0 = self.param_dict['smhm_m0_0'] + self.param_dict['smhm_m0_a']*(a - 1)
        m0 = 10.**logm0
        logm1 = self.param_dict['smhm_m1_0'] + self.param_dict['smhm_m1_a']*(a - 1)
        beta = self.param_dict['smhm_beta_0'] + self.param_dict['smhm_beta_a']*(a - 1)
        delta = self.param_dict['smhm_delta_0'] + self.param_dict['smhm_delta_a']*(a - 1)
        gamma = self.param_dict['smhm_gamma_0'] + self.param_dict['smhm_gamma_a']*(a - 1)

        stellar_mass_by_m0 = stellar_mass/m0
        term3_numerator = (stellar_mass_by_m0)**delta
        term3_denominator = 1 + (stellar_mass_by_m0)**(-gamma)

        log_halo_mass = logm1 + beta*np.log10(stellar_mass_by_m0) + (term3_numerator/term3_denominator) - 0.5

        # convert back from h=0.7 to h=1 and return the result
        return np.log10((10.**log_halo_mass)*self.littleh)

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function
        of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        redshift : float or array, optional
            Redshift of the halo hosting the galaxy. If passing an array,
            must be of the same length as the input ``stellar_mass``.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table,
            in solar mass units assuming h = 1.
        """
        redshift = safely_retrieve_redshift(self, 'mean_stellar_mass', **kwargs)

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            halo_mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            halo_mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``")

        log_stellar_mass_table = np.linspace(8.5, 12.5, 100)
        log_halo_mass_table = self.mean_log_halo_mass(log_stellar_mass_table, redshift=redshift)

        if not np.all(np.isfinite(log_halo_mass_table)):
            msg = ("The value of the mean_stellar_mass function in the Behroozi+10 model \n"
                "is calculated by numerically inverting results "
                "from the mean_log_halo_mass function.\nThese lookup tables "
                "have infinite-valued entries, which may lead to incorrect results.\n"
                "This is likely caused by the values of one or more of the model parameters "
                "being set to unphysically large/small values.")
            warn(msg)

        interpol_func = model_helpers.custom_spline(log_halo_mass_table, log_stellar_mass_table)

        log_stellar_mass = interpol_func(np.log10(halo_mass))

        stellar_mass = 10.**log_stellar_mass

        return stellar_mass
