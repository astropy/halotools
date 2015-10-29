# -*- coding: utf-8 -*-
"""

Module containing the HOD-style composite model based on Leauthaud et al. (2011).

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from ...smhm_models import Behroozi10SmHm
from ... import factories 

from ....sim_manager import sim_defaults

__all__ = ['behroozi10_model_dictionary']


def behroozi10_model_dictionary(redshift = sim_defaults.default_redshift, **kwargs):
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

    Examples 
    --------
    >>> model_blueprint = behroozi10_model_dictionary()
    >>> model_instance = factories.SubhaloModelFactory(**model_blueprint)

    """

    stellar_mass_model = Behroozi10SmHm(redshift = redshift, **kwargs)
    return {'stellar_mass': stellar_mass_model}



