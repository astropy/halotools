# -*- coding: utf-8 -*-
"""

Classes for halo profile objects. 

"""


from abc import ABCMeta, abstractmethod
import numpy as np
from occupation_helpers import aph_spline

from astropy.extern import six

from utils.array_utils import array_like_length as aph_len

import profile_components



class GalaxyProfile(object):
    """ Container class for the intra-halo phase space distribution
    of any galaxy population. 
    """

    def __init__(self,component_model_dict):
        """ The methods of this class derive their behavior from other, external classes, 
        passed in the form of the component_model_dict, a dictionary whose keys 
        are the galaxy types found in the halos, e.g., 'centrals', 'satellites', 'orphans', etc.
        The values of the component_model_dict are themselves dictionaries whose keys are 
        strings specifying the type of model being passed, e.g., 'occupation_model', and values 
        are instances of that type of model. The component_model_dict dictionary is built by 
        the hod_designer interface. 
        """
        self.component_model_dict = component_model_dict
        self.gal_types = self.component_model_dict.keys()
        self.occupation_bound = {}
        for gal_type in self.gal_types:
            self.occupation_bound[gal_type] = (
                self.component_model_dict[gal_type]['occupation_model'].upper_bound)

        self.parameter_dict = (
            self.retrieve_all_inherited_parameters(
                self.component_model_dict)
            )
        self.publications = []







