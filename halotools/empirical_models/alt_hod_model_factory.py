# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from functools import partial

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import model_helpers as model_helpers
from . import model_defaults
from . import mock_factories
from . import preloaded_hod_blueprints
from . import gal_prof_factory
from . import halo_prof_components

from ..sim_manager.supported_sims import HaloCatalog

from ..sim_manager.generate_random_sim import FakeSim
from ..utils.array_utils import custom_len

from ..custom_exceptions import HalotoolsError
from warnings import warn 

from .model_factories import ModelFactory

class AltHodModelFactory(ModelFactory):
    """ Class used to build HOD-style models of the galaxy-halo connection. 

    Can be thought of as a factory that takes an HOD model blueprint as input, 
    and generates an HOD Model object. The returned object can be used directly to 
    populate a simulation with a Monte Carlo realization of the model. 

    Most behavior is derived from external classes bound up in the input ``model_blueprint``. 
    So the purpose of `HodModelFactory` is mostly to compose these external 
    behaviors together into a composite model. 
    The aim is to provide a standardized model object 
    that interfaces consistently with the rest of the package, 
    regardless of the features of the model. 

    Notes 
    -----
    There are two main options for creating HOD-style blueprints 
    that can be passed to this class:

        * You can use one of the pre-computed blueprint found in `~halotools.empirical_models.preloaded_hod_blueprints` 
    
    """

    def __init__(self, input_model_blueprint, **kwargs):
        """
        Parameters
        ----------
        input_model_blueprint : dict 
            The main dictionary keys of ``input_model_blueprint`` 
            are the names of the types of galaxies 
            found in the halos, 
            e.g., ``centrals``, ``satellites``, ``orphans``, etc. 
            The dictionary value associated with each ``gal_type`` key 
            is itself a dictionary whose keys 
            specify the type of model component, e.g., ``occupation``, 
            and values are class instances of that type of model. 
            The `interpret_input_model_blueprint` translates 
            ``input_model_blueprint`` into ``self.model_blueprint``.

        """

        super(HodModelFactory, self).__init__(input_model_blueprint, **kwargs)

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()
        self.model_blueprint = self._input_model_blueprint
        self._test_blueprint_consistency()

        # Build the composite model dictionary, 
        # whose keys are parameters of our model
        self._set_init_param_dict()

        # Build up and bind several lists from the component models
        self._build_composite_lists()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()


    def _set_gal_types(self):
        """ Private method binding the ``gal_types`` list attribute. 
        If there are both centrals and satellites, method ensures that centrals 
        will always be built first, out of consideration for satellite 
        model components with explicit dependence on the central population. 
        """
        gal_types = [key for key in self._input_model_blueprint.keys() if key is not 'mock_factory']
        if len(gal_types) == 1:
            self.gal_types = gal_types
        elif len(gal_types) == 2:
            self.gal_types = ['centrals', 'satellites']
        else:
            raise HalotoolsError("The HOD _input_model_blueprint currently only permits "
                "gal_types = 'centrals' and 'sateliltes'")

        for gal_type in self.gal_types:
            if gal_type not in self._input_model_blueprint.keys():
                raise HalotoolsError("The HOD _input_model_blueprint currently only permits "
                    "gal_types = 'centrals' and 'sateliltes'")



    def _set_primary_behaviors(self):
        """ Creates names and behaviors for the primary methods of `HodModelFactory` 
        that will be used by the outside world.  

        Notes 
        -----
        The new methods created here are given standardized names, 
        for consistent communication with the rest of the package. 
        This consistency is particularly important for mock-making, 
        so that the `HodMockFactory` can always call the same functions 
        regardless of the complexity of the model. 

        The behaviors of the methods created here are defined elsewhere; 
        `_set_primary_behaviors` just creates a symbolic link to those external behaviors. 
        """

        for gal_type in self.gal_types:

            ###########################
            # Set the method used to return Monte Carlo realizations 
            # of per-halo gal_type abundance
            occupation_model = self.model_blueprint[gal_type]['occupation']
            self.threshold = occupation_model.threshold

            new_method_name = 'mc_occupation_'+gal_type
            new_method_behavior = self._update_param_dict_decorator(
                gal_type, 'occupation', 'mc_occupation')
            setattr(self, new_method_name, new_method_behavior)
            
            ###########################
            # Set any additional methods requested by the component models
            if hasattr(occupation_model, '_additional_methods_to_inherit'):
                additional_methods_to_inherit = list(
                    set(occupation_model._additional_methods_to_inherit))
                for methodname in additional_methods_to_inherit:
                    new_method_name = methodname + '_' + gal_type
                    new_method_behavior = self._update_param_dict_decorator(
                        gal_type, 'occupation', methodname)
                    setattr(self, new_method_name, new_method_behavior)

            ###########################
            # Set the method used to assign positions and velocities
            gal_prof_model = self.model_blueprint[gal_type]['profile']
            for prof_param_key in gal_prof_model.prof_param_keys:

            # Create a new method to compute each (unbiased) halo profile parameter
                new_method_name = prof_param_key + '_halos'
                # For composite models in which multiple galaxy types have the same 
                # underlying dark matter profile, use the halo profile model of the 
                # first gal_type in the self.gal_types list 
                if not hasattr(self, new_method_name):
                    new_method_behavior = getattr(gal_prof_model, prof_param_key)
                    setattr(self, new_method_name, new_method_behavior)

            setattr(self, 'assign_phase_space_' + gal_type, 
                gal_prof_model.assign_phase_space)



    def _update_param_dict_decorator(self, gal_type, component_key, func_name):
        """ Decorator used to propagate any possible changes 
        in the composite model param_dict 
        down to the appropriate component model param_dict. 
        """

        component_model = self.model_blueprint[gal_type][component_key]

        def decorated_func(*args, **kwargs):

            # Update the param_dict as necessary
            for key in self.param_dict.keys():
                if key in component_model.param_dict:
                    component_model.param_dict[key] = self.param_dict[key]

            func = getattr(component_model, func_name)
            return func(*args, **kwargs)

        return decorated_func

    def build_lookup_tables(self):
        """ Method to compute and load lookup tables for each of 
        the phase space component models. 
        """

        for gal_type in self.gal_types:
            self.model_blueprint[gal_type]['profile'].build_lookup_tables()

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

        def test_expected_key_repetition(model, key):
            if hasattr(model, 'ancillary_model_param_keys'):
                if key in model.ancillary_model_param_keys:
                    return 
                    
            raise HalotoolsError("The param_dict key %s appears in more "
                "than one component model" % key)

        # Loop over all galaxy types in the composite model
        for gal_type in self.gal_types:
            gal_type_dict = self.model_blueprint[gal_type]
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():

                intersection = set(self.param_dict) & set(model_instance.param_dict)
                if intersection != set():
                    for key in intersection:
                        test_expected_key_repetition(model_instance, key)

                for key, value in model_instance.param_dict.iteritems():
                    self.param_dict[key] = value

        self._init_param_dict = copy(self.param_dict)

    def restore_init_param_dict(self):
        """ Reset all values of the current ``param_dict`` to the values 
        the class was instantiated with. 

        Primary behaviors are reset as well, as this is how the 
        inherited behaviors get bound to the values in ``param_dict``. 
        """
        self.param_dict = self._init_param_dict
        self._set_primary_behaviors()

    def _build_composite_lists(self):
        """ A composite model has several lists that are built up from 
        the components: ``_haloprop_list``, ``publications``, and 
        ``new_haloprop_func_dict``. 
        """

        haloprop_list = []
        prof_param_keys = []
        pub_list = []
        new_haloprop_func_dict = {}

        for gal_type in self.gal_types:
            component_dict = self.model_blueprint[gal_type]
            for component_key in component_dict.keys():
                component_model = component_dict[component_key]

                # haloprop keys
                if hasattr(component_model, 'prim_haloprop_key'):
                    haloprop_list.append(component_model.prim_haloprop_key)
                if hasattr(component_model, 'sec_haloprop_key'):
                    haloprop_list.append(component_model.sec_haloprop_key)

                # halo profile parameter keys
                if hasattr(component_model, 'prof_param_keys'):
                    prof_param_keys.extend(component_model.prof_param_keys)

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
                        msg = ("The composite model received multiple "
                            "component models \nwith a new_haloprop_func_dict that use "
                            "the %s key. \nIgnoring the one that appears in the %s " 
                            "component for %s galaxies")
                        warn(msg % (example_repeated_element, component_key, gal_type))

        self._haloprop_list = list(set(haloprop_list))
        self.prof_param_keys = list(set(prof_param_keys))
        self.publications = list(set(pub_list))
        self.new_haloprop_func_dict = new_haloprop_func_dict

    def _test_blueprint_consistency(self):
        """
        Method tests to make sure that all HOD occupation components have the same 
        threshold, and raises an exception if not. 
        """
        threshold_list = []
        threshold_msg = ''
        for gal_type in self.gal_types:
            component_dict = self.model_blueprint[gal_type]
            for component_key in component_dict.keys():
                component_model = component_dict[component_key]
                if component_key == 'occupation':
                    threshold_list.append(component_model.threshold)
                    threshold_msg = threshold_msg + '\n' + gal_type + ' threshold = ' + str(component_model.threshold)
        if len(threshold_list) > 1:
            d = np.diff(threshold_list)
            if np.any(d != 0):
                msg = ("Inconsistency in the threshold of the component occupation models:\n" + threshold_msg + "\n")
                raise HalotoolsError(msg)



##########################################











