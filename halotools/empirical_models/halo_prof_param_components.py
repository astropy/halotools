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
        prim_haloprop_key = model_defaults.prim_haloprop_key):

        self.cosmology = cosmology
        self.redshift = redshift
        self.prim_haloprop_key = prim_haloprop_key

#    def conc_mass(self, mass, model=model_defaults.conc_mass_relation_key, **kwargs):
    def conc_mass(self, **kwargs):
        """ Method used to evaluate the mean NFW concentration as a function of 
        halo mass. 

        This is the primary method seen by the outside world. It has no functionality 
        of its own, it only calls the desired function based on the model keyword. 

        Parameters 
        ----------
        mass : array_like 

        model : string, optional 
            Used to specify which model to use for the concentration-mass relation. 
            Currently supported relations are 'dutton_maccio14'. 

        redshift : float or array_like, optional keyword argument
            If redshift is an array, must be same length as mass. 
            If no redshift keyword is passed, model_defaults.default_redshift will be chosen. 

        """

        if 'redshift' in kwargs.keys():
            z = kwargs['redshift']
        else:
            z = self.redshift

        if 'mass' in kwargs.keys():
            mass = kwargs['mass']
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'galaxy_table' in kwargs.keys():
            halo_mass_key = model_defaults.host_haloprop_prefix + self.prim_haloprop_key
            mass = kwargs['galaxy_table'][halo_mass_key]

        if 'model' not in kwargs.keys():
            model = model_defaults.conc_mass_relation_key
        else:
            model = kwargs['model']

        if model == 'dutton_maccio14':
            return self.dutton_maccio14_conc_mass(mass, z)
        else:
            raise KeyError("Input conc-mass model is not supported. "
                "The only currently supported conc-mass model is dutton_maccio14")


    def dutton_maccio14_conc_mass(self, mass, z):
        """ Power-law fit to the concentration-mass relation from 
        Equations 12 & 13 of Dutton & Maccio 2014, arXiv:1402.7073.

        Parameters 
        ----------
        mass : array_like 

        z : float or array_like 
            If z is an array, must be same length as mass

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

        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
        b = -0.097 + 0.024 * z
        m0 = 1.e12

        logc = a + b * np.log10(mass / m0)
        c = 10**logc

        return c
