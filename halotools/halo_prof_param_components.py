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
import defaults

class ConcMass(object):
    """ Container class for commonly used concentration-mass 
    relations in the literature. 
    """

    def __init__(self, cosmology=defaults.default_cosmology, 
        redshift = defaults.default_redshift):
        self.cosmology = cosmology
        self.redshift = redshift

    def conc_mass(self, mass, model='dutton_maccio14', **kwargs):
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
            If no redshift keyword is passed, defaults.default_redshift will be chosen. 

        """

        if 'redshift' in kwargs.keys():
            z = kwargs['redshift']
        else:
            z = self.redshift

        if model == 'dutton_maccio14':
            return self.dutton_maccio14_conc_mass(mass, z)
        else:
            raise KeyError("input conc-mass model is not supported")


    def dutton_maccio14_conc_mass(self, mass, z):
        """ Power-law fit to the concentration-mass relation from 
        Dutton & Maccio 2014, MNRAS 441, 3359, arXiv:1402.7073.

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
        """

        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
        b = -0.097 + 0.024 * z
        m0 = 1.e12

        logc = a + b * np.log10(mass / m0)
        c = 10**logc

        return c
