# -*- coding: utf-8 -*-
"""

Module containing the HOD-style composite model based on Leauthaud et al. (2011).

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from ... import factories
from ...smhm_models import Behroozi10SmHm

from ....sim_manager import sim_defaults

__all__ = ['Behroozi10']

def Behroozi10(redshift = sim_defaults.default_redshift, **kwargs):
	"""
    Parameters 
    ----------
    redshift : float, optional 
    	Redshift of the stellar-to-halo-mass relation of the model. Must be consistent 
    	with the redshift of the halo catalog you populate. Default value is set by 
    	sim_defaults.default_redshift. 

    prim_haloprop_key : string, optional  
        String giving the column name of the primary halo property governing stellar mass. 
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    scatter_model : object, optional  
        Class governing stochasticity of stellar mass. Default scatter is log-normal, 
        implemented by the `LogNormalScatterModel` class. 

    scatter_abcissa : array_like, optional  
        Array of values giving the abcissa at which
        the level of scatter will be specified by the input ordinates.
        Default behavior will result in constant scatter at a level set in the 
        `~halotools.empirical_models.model_defaults` module. 

    scatter_ordinates : array_like, optional  
        Array of values defining the level of scatter at the input abcissa.
        Default behavior will result in constant scatter at a level set in the 
        `~halotools.empirical_models.model_defaults` module. 
	"""

	stellar_mass_model = Behroozi10SmHm(redshift = redshift, **kwargs)
	model_blueprint = {'stellar_mass': stellar_mass_model}
	composite_model = factories.SubhaloModelFactory(model_blueprint)

	return composite_model


