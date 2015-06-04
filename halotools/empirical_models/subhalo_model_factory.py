# -*- coding: utf-8 -*-
"""
Temporary module storing the SubhaloModelFactory. 
This module exists for development purposes only - 
eventually its contents will be cannibalized by a general model factory module. 
"""

__all__ = ['SubhaloModelFactory']
__author__ = ['Andrew Hearin']

from functools import partial 

import numpy as np

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import occupation_helpers as occuhelp
from . import model_defaults
from . import mock_factory
from ..utils.array_utils import array_like_length as aph_len

class SubhaloModelFactory(object):
    """ Class used to build any model of the galaxy-halo connection 
    in which there is a one-to-one correspondence between subhalos and galaxies.  

    Can be thought of as a factory that takes a model blueprint as input, 
    and generates a Subhalo Model object. The returned object can be used directly to 
    populate a simulation with a Monte Carlo realization of the model. 
    """

    def __init__(self, input_model_blueprint, **kwargs):
        """
        Parameters
        ----------
        input_model_blueprint : dict 
            The main dictionary keys of ``input_model_blueprint`` 
            are ``galprop_key`` strings, the names of 
            properties that will be assigned to galaxies 
            e.g., ``stellar_mass``, ``sfr``, ``morphology``, etc. 
            The dictionary value associated with each ``galprop_key``  
            is a class instance of the type of model that 
            maps that property onto subhalos. 

        """

        # Bind the model-building instructions to the composite model
        self.model_blueprint = input_model_blueprint

        if 'mock_factory' not in self.model_blueprint.keys():
            self.model_blueprint['mock_factory'] = mock_factory.SubhaloMockFactory

        galprop_key_list = [key for key in self.model_blueprint.keys() if key is not 'mock_factory']
        self.galprop_keys = galprop_key_list
        # galprop_keys should be an ordered list of strings of the properties to assign 
