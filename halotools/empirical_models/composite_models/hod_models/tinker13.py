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
    # Build the occupation model
    centrals_occupation = tinker13_components.Tinker13Cens(threshold = threshold, **kwargs)
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model
    
    centrals_profile = TrivialPhaseSpace(velocity_bias = central_velocity_bias, **kwargs)

    
    # Build the occupation model
    quiescent_satellites_occupation = tinker13_components.Tinker13QuiescentSats(threshold = threshold, **kwargs)
    # Build the profile model
    quiescent_satellites_profile = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)    

    # Build the occupation model
    active_satellites_occupation = tinker13_components.Tinker13ActiveSats(threshold = threshold, **kwargs)
    # Build the profile model
    active_satellites_profile = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)  
    del active_satellites_profile.new_haloprop_func_dict
    
    composite_model = factories.HodModelFactory(centrals_occupation = centrals_occupation, 
        centrals_profile = centrals_profile, quiescent_satellites_profile = quiescent_satellites_profile, 
        quiescent_satellites_occupation = quiescent_satellites_occupation, 
        active_satellites_profile = active_satellites_profile, 
        active_satellites_occupation = active_satellites_occupation, 
        gal_type_list = ['centrals', 'active_satellites', 'quiescent_satellites'])
    
    return composite_model











