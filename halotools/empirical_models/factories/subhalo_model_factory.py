# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__all__ = ['SubhaloModelFactory']
__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from functools import partial
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from warnings import warn 

from .model_factory_template import ModelFactory
from .hod_mock_factory import HodMockFactory
from .subhalo_mock_factory import SubhaloMockFactory

from .. import model_helpers
from .. import model_defaults 

from ...sim_manager.supported_sims import HaloCatalog
from ...sim_manager import sim_defaults
from ...sim_manager.generate_random_sim import FakeSim
from ...utils.array_utils import custom_len
from ...custom_exceptions import *



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

        galprop_sequence : list, optional
            Some model components may have explicit dependence upon 
            the value of some other galaxy model property. A classic 
            example is if the stellar mass of a central galaxy has explicit 
            dependence on whether or not the central is active or quiescent. 
            In such a case, you must pass a list of the galaxy properties 
            of the composite model; the first galprop in ``galprop_sequence`` 
            will be assigned first by the ``mock_factory``; the second galprop 
            in ``galprop_sequence`` will be assigned second, and its computation 
            may depend on the first galprop, and so forth. Default behavior is 
            to assume that no galprop has explicit dependence upon any other. 

        galaxy_selection_func : function object, optional  
            Function object that imposes a cut on the mock galaxies. 
            Function should take an Astropy table as a positional argument, 
            and return a boolean numpy array that will be 
            treated as a mask over the rows of the table. If not None, 
            the mask defined by ``galaxy_selection_func`` will be applied to the 
            ``halo_table`` after the table is generated by the `populate_mock` method. 
            Default is None.  

        halo_selection_func : function object, optional   
            Function object used to place a cut on the input ``snapshot.halo_table`` table. 
            Default behavior depends on the sub-class of `MockFactory`. 
            If the ``halo_selection_func`` keyword argument is passed, 
            the input to the function must be a length-Nsubhalos structured numpy array or Astropy table; 
            the function output must be a length-Nsubhalos boolean array that will be used as a mask. 
        """

        super(SubhaloModelFactory, self).__init__(input_model_blueprint, **kwargs)

        self.mock_factory = SubhaloMockFactory

        self.model_blueprint = copy(self._input_model_blueprint)
        
        self._build_composite_attrs(**kwargs)

        self._set_init_param_dict()

        self._set_primary_behaviors()

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

        for galprop_key in self.galprop_list:
            
            behavior_name = 'mc_'+galprop_key
            behavior_function = self._update_param_dict_decorator(galprop_key, behavior_name)
            setattr(self, behavior_name, behavior_function)

    def _update_param_dict_decorator(self, galprop_key, func_name):
        """ Decorator used to propagate any possible changes 
        in the composite model param_dict 
        down to the appropriate component model param_dict. 
        """

        component_model = self.model_blueprint[galprop_key]

        def decorated_func(*args, **kwargs):

            # Update the param_dict as necessary
            for key in component_model.param_dict.keys():
                composite_key = galprop_key + '_' + key
                if composite_key in self.param_dict.keys():
                    component_model.param_dict[key] = self.param_dict[composite_key]

            # # Also update the param dict of ancillary models, if applicable
            # if hasattr(component_model, 'ancillary_model_dependencies'):
            #     for model_name in component_model.ancillary_model_dependencies:

            #         dependent_galprop_key = getattr(component_model, model_name).galprop_key
            #         for key in getattr(component_model, model_name).param_dict.keys():
            #             composite_key = composite_key = dependent_galprop_key + '_' + key
            #             if composite_key in self.param_dict.keys():
            #                 getattr(component_model, model_name).param_dict[key] = (
            #                     self.param_dict[composite_key]
            #                     )

            func = getattr(component_model, func_name)
            return func(*args, **kwargs)

        return decorated_func

    def _galprop_func(self, galprop_key):
        """
        """
        component_model = self.model_blueprint[galprop_key]
        behavior_function = getattr(component_model, 'mc_'+galprop_key) 
        return behavior_function

    def _build_composite_attrs(self, **kwargs):
        """ A composite model has several bookkeeping devices that are built up from 
        the components: ``_haloprop_list``, ``publications``, and ``new_haloprop_func_dict``. 
        """

        unordered_galprop_list = [key for key in self.model_blueprint.keys()]
        if 'galprop_sequence' in kwargs.keys():
            if set(kwargs['galprop_sequence']) != set(unordered_galprop_list):
                raise KeyError("The input galprop_sequence keyword argument must "
                    "have the same list of galprops as the input model blueprint")
            else:
                self.galprop_list = kwargs['galprop_sequence']
        else:
            self.galprop_list = unordered_galprop_list

        haloprop_list = []
        pub_list = []
        new_haloprop_func_dict = {}

        for galprop in self.galprop_list:
            component_model = self.model_blueprint[galprop]

            # haloprop keys
            if hasattr(component_model, 'prim_haloprop_key'):
                haloprop_list.append(component_model.prim_haloprop_key)
            if hasattr(component_model, 'sec_haloprop_key'):
                haloprop_list.append(component_model.sec_haloprop_key)

            # Reference list
            if hasattr(component_model, 'publications'):
                pub_list.extend(component_model.publications)

            # Haloprop function dictionaries
            if hasattr(component_model, 'new_haloprop_func_dict'):
                dict_intersection = set(new_haloprop_func_dict).intersection(
                    set(component_model.new_haloprop_func_dict))
                if dict_intersection == set():
                    new_haloprop_func_dict = dict(
                        new_haloprop_func_dict.items() + 
                        component_model.new_haloprop_func_dict.items()
                        )
                else:
                    example_repeated_element = list(dict_intersection)[0]
                    raise KeyError("The composite model received multiple "
                        "component models with a new_haloprop_func_dict that use "
                        "the %s key" % example_repeated_element)

        self._haloprop_list = list(set(haloprop_list))
        self.publications = list(set(pub_list))
        self.new_haloprop_func_dict = new_haloprop_func_dict

    def _set_init_param_dict(self):
        """ Method used to build a dictionary of parameters for the composite model. 

        Accomplished by retrieving all the parameters of the component models. 
        Method returns nothing, but binds ``param_dict`` to the class instance. 

        Notes 
        -----
        In MCMC applications, the items of ``param_dict`` define the 
        parameter set explored by the likelihood engine. 
        Changing the values of the parameters in ``param_dict`` 
        will propagate to the behavior of the component models. 

        Each component model has its own ``param_dict`` bound to it. 
        When changing the values of ``param_dict`` bound to `HodModelFactory`, 
        the corresponding values of the component model ``param_dict`` will *not* change.  

        """

        self.param_dict = {}

        # Loop over all galaxy types in the composite model
        for galprop in self.galprop_list:
            galprop_model = self.model_blueprint[galprop]

            if hasattr(galprop_model, 'param_dict'):
                galprop_model_param_dict = (
                    {galprop_model.galprop_key+'_'+key:val for key, val in galprop_model.param_dict.items()}
                    )
            else:
                galprop_model_param_dict = {}

            intersection = set(self.param_dict) & set(galprop_model_param_dict)
            if intersection != set():
                repeated_key = list(intersection)[0]
                raise KeyError("The param_dict key %s appears in more "
                    "than one component model" % repeated_key)
            else:

                self.param_dict = dict(
                    galprop_model_param_dict.items() + 
                    self.param_dict.items()
                    )

        self._init_param_dict = copy(self.param_dict)

    def restore_init_param_dict(self):
        """ Reset all values of the current ``param_dict`` to the values 
        the class was instantiated with. 

        Primary behaviors are reset as well, as this is how the 
        inherited behaviors get bound to the values in ``param_dict``. 
        """
        self.param_dict = self._init_param_dict
        self._set_primary_behaviors()
