# -*- coding: utf-8 -*-
"""

halotools.halo_prof_param_components contains the classes and functions 
defining relations between dark matter halos and the parameters 
governing their internal structure. The classic example of such a component 
is a relation between halo mass and NFW concentration. More generally, 
a halo profile parameter model is just a mapping between *any* 
halo property and *any* profile parameter. 

"""

import numpy as np
import model_defaults
from ..sim_manager import sim_defaults
import model_defaults

__all__ = ['ConcMass']

class ConcMass(object):
    """ Container class for commonly used concentration-mass 
    relations in the literature. 

    For all models, cosmology-dependent quantities such as peak height 
    are solved for using the Astropy `~astropy.cosmology` sub-package. 

    Currently supported fitting functions include:

        * `dutton_maccio14_conc_mass`

    """

    def __init__(self, cosmology=sim_defaults.default_cosmology, 
        redshift = sim_defaults.default_redshift, 
        prim_haloprop_key = model_defaults.prim_haloprop_key, 
        conc_mass_model = model_defaults.conc_mass_model):

        self.cosmology = cosmology
        self.redshift = redshift
        self.prim_haloprop_key = prim_haloprop_key
        self.conc_mass_model = conc_mass_model

    def __call__(self, **kwargs):
        """ Method used to evaluate the mean NFW concentration as a function of 
        halo mass. 

        This is the primary method seen by the outside world. It has no functionality 
        of its own, it only calls the desired function based on the model keyword. 

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument
            Array storing a mass-like variable that governs the occupation statistics. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing mock galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        """
        # Retrieve the array storing the mass-like variable
        if 'galaxy_table' in kwargs.keys():
            key = model_defaults.host_haloprop_prefix+self.prim_haloprop_key
            mass = kwargs['galaxy_table'][key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``prim_haloprop``, or ``galaxy_table``")

        return getattr(self, self.conc_mass_model)(mass)

    def dutton_maccio14(self, mass):
        """ Power-law fit to the concentration-mass relation from 
        Equations 12 & 13 of Dutton & Maccio 2014, arXiv:1402.7073.

        Parameters 
        ----------
        mass : array_like 

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

        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * self.redshift**1.08)
        b = -0.097 + 0.024 * self.redshift
        m0 = 1.e12

        logc = a + b * np.log10(mass / m0)
        c = 10**logc

        return c
