# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from ... import factories, model_defaults
from ...occupation_models import leauthaud11_components 
from ...occupation_models import tinker13_components 

from ...smhm_models import Behroozi10SmHm
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

from ....sim_manager import FakeSim, sim_defaults


__all__ = ['Tinker13']

def Tinker13(threshold = model_defaults.default_stellar_mass_threshold, 
    central_velocity_bias = False, satellite_velocity_bias = False, **kwargs):
    """
    """
    cen_key = 'centrals'
    subpopulation_blueprint_centrals = {}
    # Build the occupation model
    occupation_feature_centrals = tinker13_components.Tinker13Cens(threshold = threshold, **kwargs)
    occupation_feature_centrals._suppress_repeated_param_warning = True
    subpopulation_blueprint_centrals['occupation'] = occupation_feature_centrals
    # Build the profile model
    
    profile_feature_centrals = TrivialPhaseSpace(velocity_bias = central_velocity_bias, **kwargs)

    subpopulation_blueprint_centrals['profile'] = profile_feature_centrals
    
    sat_key1 = 'quiescent_satellites'
    subpopulation_blueprint_satellites1 = {}
    # Build the occupation model
    occupation_feature_satellites1 = tinker13_components.Tinker13QuiescentSats(threshold = threshold, **kwargs)
    subpopulation_blueprint_satellites1['occupation'] = occupation_feature_satellites1
    # Build the profile model
    profile_feature_satellites1 = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)    
    subpopulation_blueprint_satellites1['profile'] = profile_feature_satellites1

    sat_key2 = 'active_satellites'
    subpopulation_blueprint_satellites2 = {}
    # Build the occupation model
    occupation_feature_satellites2 = tinker13_components.Tinker13ActiveSats(threshold = threshold, **kwargs)
    subpopulation_blueprint_satellites2['occupation'] = occupation_feature_satellites2
    # Build the profile model
    profile_feature_satellites2 = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)  
    del profile_feature_satellites2.new_haloprop_func_dict
    subpopulation_blueprint_satellites2['profile'] = profile_feature_satellites2
    
    blueprint = {cen_key: subpopulation_blueprint_centrals, 
                 sat_key1: subpopulation_blueprint_satellites1, 
                 sat_key2: subpopulation_blueprint_satellites2}
    
    return factories.HodModelFactory(blueprint)











