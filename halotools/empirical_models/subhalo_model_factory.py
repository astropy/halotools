# -*- coding: utf-8 -*-
"""
Temporary module storing the SubhaloModelFactory. 
This module exists for development purposes only - 
eventually its contents will be cannibalized by a general model factory module. 
"""

__all__ = ['ModelFactory', 'SubhaloModelFactory']
__author__ = ['Andrew Hearin']

from functools import partial 

import numpy as np
from copy import copy

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import occupation_helpers as occuhelp
from . import model_defaults
from . import mock_factory

from ..utils.array_utils import array_like_length as aph_len
from ..sim_manager.read_nbody import ProcessedSnapshot

@six.add_metaclass(ABCMeta)
class ModelFactory(object):
    """ Abstract container class used to build 
    any composite model of the galaxy-halo connection. 
    """

    def __init__(self, input_model_blueprint, **kwargs):

        # Bind the model-building instructions to the composite model
        self._input_model_blueprint = input_model_blueprint

    def populate_mock(self, **kwargs):
        """ Method used to populate a simulation using the model. 

        After calling this method, ``self`` will have a new ``mock`` attribute, 
        which has a ``galaxy_table`` bound to it containing the Monte Carlo 
        realization of the model. 

        Parameters 
        ----------
        snapshot : object, optional keyword argument
            Class instance of `~halotools.sim_manager.ProcessedSnapshot`. 
            This object contains the halo catalog and its metadata.  

        """

        if hasattr(self, 'mock'):
            self.mock.populate()
        else:
            if 'snapshot' in kwargs.keys():
                snapshot = kwargs['snapshot']
                # we need to delete the 'snapshot' keyword 
                # or else the call to mock_factory below 
                # will pass multiple snapshot arguments
                del kwargs['snapshot']
            else:
                snapshot = ProcessedSnapshot(**kwargs)

            mock_factory = self.model_blueprint['mock_factory']
            mock = mock_factory(snapshot, self, **kwargs)
            self.mock = mock



class SubhaloModelFactory(ModelFactory):
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

        super(SubhaloModelFactory, self).__init__(input_model_blueprint, **kwargs)

        self.model_blueprint, self.galprop_list = self._interpret_input_model_blueprint()

        self._set_primary_behaviors()


    def _interpret_input_model_blueprint(self):

        model_blueprint = copy(self._input_model_blueprint)

        if 'mock_factory' not in model_blueprint.keys():
            model_blueprint['mock_factory'] = mock_factory.SubhaloMockFactory

        unordered_galprop_list = [key for key in model_blueprint.keys() if key is not 'mock_factory']

        # If necessary, put the unordered_galprop_list into its proper order
        # Note that this is only robust to the case of two-property composite models
        # For more complicated models, a smarter algorithm will be necessary, 
        # so we raise an exception to protect against that case
        if aph_len(unordered_galprop_list) > 2:
            raise KeyError("SubhaloModelFactory does not support assignment of "
                "more than two galaxy properties")

        temp_required_galprop_dict = {}
        for galprop in unordered_galprop_list:
            component_model = model_blueprint[galprop]
            if hasattr(component_model, 'required_galprops'):
                temp_required_galprop_dict[galprop] = component_model.required_galprops

        if len(temp_required_galprop_dict) == 0:
            galprop_list = unordered_galprop_list
            
        elif len(temp_required_galprop_dict) == 1:
            galprop_list = temp_required_galprop_dict.values()[0]
            galprop_list.extend(temp_required_galprop_dict.keys()[0])
        else:
            raise KeyError("Cannot resolve model interdependencies:\n"
                "Both component models depend on each other simultaneously\n"
                "This composite model cannot be decomposed in a sensible way")

        return model_blueprint, galprop_list

    def _set_primary_behaviors(self):
        """ Creates names and behaviors for the primary methods of `SubhaloModelFactory` 
        that will be used by the outside world.  

        Notes 
        -----
        The new methods created here are given standardized names, 
        for consistent communication with the rest of the package. 
        This consistency is particularly important for mock-making, 
        so that the `SubhaloModelFactory` can always call the same functions 
        regardless of the complexity of the model. 

        The behaviors of the methods created here are defined elsewhere; 
        `_set_primary_behaviors` just creates a symbolic link to those external behaviors. 
        """

        for galprop in self.galprop_list:
            component_model = self.model_blueprint[galprop]
            new_method_name = galprop + '_model_func'
            new_method_behavior = component_model.__call__
            setattr(self, new_method_name, new_method_behavior)




















