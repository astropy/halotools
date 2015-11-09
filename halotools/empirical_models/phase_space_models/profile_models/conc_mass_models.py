# -*- coding: utf-8 -*-
"""

Classes and functions defining relations between NFW concentration 
and halo mass. 

"""

import numpy as np

from ... import model_defaults
from ...model_helpers import bind_default_kwarg_mixin_safe

from ....sim_manager import sim_defaults
from ....custom_exceptions import HalotoolsError

__all__ = ['ConcMass']

class ConcMass(object):
    """ Container class for commonly used concentration-mass relations in the literature. 

    `ConcMass` can be instantiated as a stand-alone class, or used as an orthogonal mix-in 
    with the `~halotools.empirical_models.NFWProfile` or any of its sub-classes. 

    Notes 
    ------
    The only currently supported model is `dutton_maccio14`.

    """

    def __init__(self, conc_mass_model = model_defaults.conc_mass_model, **kwargs):
        """
        Parameters 
        ----------
        cosmology : object, optional 
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        mdef: str, optional 
            String specifying the halo mass definition, e.g., 'vir' or '200m'. 
             Default is set in `~halotools.empirical_models.model_defaults`.

        conc_mass_model : string, optional  
            Specifies the calibrated fitting function used to model the concentration-mass relation. 
             Default is set in `~halotools.empirical_models.model_defaults`.

        concentration_key : string, optional 
            Key of the ``halo_table`` column storing the halo catalog concentration. 
            The only concentration-mass model requiring this constructor input is the 
            ``direct_from_halo_catalog`` model. Default is ``halo_nfw_conc``, which is consistent 
            with all halo catalogs provided by Halotools but may differ from the convention
            adopted in custom halo catalogs. 

        Examples 
        ---------
        >>> conc_mass_model = ConcMass()
        >>> conc_mass_model = ConcMass(redshift = 2, mdef = '500c')

        """
        self.conc_mass_model = conc_mass_model

        bind_default_kwarg_mixin_safe(self, 'cosmology', kwargs, sim_defaults.default_cosmology)
        bind_default_kwarg_mixin_safe(self, 'redshift', kwargs, sim_defaults.default_redshift)
        bind_default_kwarg_mixin_safe(self, 'mdef', kwargs, model_defaults.halo_mass_definition)

        if not hasattr(self, 'halo_mass_key'):
            self.halo_mass_key = model_defaults.get_halo_mass_key(self.mdef)

        if 'concentration_key' in kwargs:
            self.concentration_key = kwargs['concentration_key']
        else:
            self.concentration_key = model_defaults.concentration_key

    def compute_concentration(self, **kwargs):
        """ Method used to evaluate the mean NFW concentration as a function of 
        halo mass. 

        This is the primary method seen by the outside world. It has no functionality 
        of its own, it only calls the desired function based on the model keyword. 

        Parameters
        ----------        
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 

        Notes 
        -----
        The testing for this model can be found in 
        `~halotools.empirical_models.test_empirical_models.test_conc_mass`. 

        """
        conc_mass_func = getattr(self, self.conc_mass_model)
        return conc_mass_func(**kwargs)

    def direct_from_halo_catalog(self, **kwargs):
        """
        Parameters 
        ----------
        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 
        """
        key = self.concentration_key

        try:
            c = kwargs['halo_table'][key]
        except KeyError:
            msg = ("\nIf you are using the ``direct_from_halo_catalog`` concentration-mass model,\n"
                "you must pass a ``concentration_key`` keyword argument to the ConcMass constructor\n"
                "that corresponds to a key in the input ``halo_table``. \n"
                "For all Halotools-provided processed halo catalogs, the appropriate key is ``halo_nfw_conc``,\n"
                "which is the default choice for this model.\n"
                "However, if you are using a halo catalog of your own, \n"
                "you must make sure that you specify the appropriate ``concentration_key`` for your catalog.\n"
                "The current ConcMass instance is set to use %s as the ``concentration_key``, \n"
                "which does not correspond to a column of the input ``halo_table``.\n")
            raise HalotoolsError(msg % key)
        
        ### Impose the model_defaults constraints
        c = np.where(c > model_defaults.max_permitted_conc, 
            model_defaults.max_permitted_conc, c)
        c = np.where(c < model_defaults.min_permitted_conc, 
            model_defaults.min_permitted_conc, c)

        return c

    def dutton_maccio14(self, **kwargs):
        """ Power-law fit to the concentration-mass relation from 
        Equations 12 & 13 of Dutton & Maccio 2014, arXiv:1402.7073.

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 

        Notes 
        -----        
        This model was only calibrated for the Planck 1-year cosmology.

        Model assumes that halo mass definition is Mvir.

        :math:`a = 0.537 + (1.025 - 0.537)\\exp(-0.718z^{1.08})`

        :math:`b = -0.097 + 0.024z`

        :math:`M_{0} = 10^{12}M_{\odot}/h`

        :math:`\\log_{10}c(M) = a + b\\log_{10}(M / M_{0})`

        """

        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            mass = kwargs['halo_table'][self.halo_mass_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments "
                "to the compute_concentration method:\n"
                "``halo_table`` or ``prim_haloprop``")

        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * self.redshift**1.08)
        b = -0.097 + 0.024 * self.redshift
        m0 = 1.e12

        logc = a + b * np.log10(mass / m0)
        c = 10**logc

        return c
