# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy

from . import model_helpers, mock_factories

from ..sim_manager.supported_sims import HaloCatalog
from ..sim_manager.generate_random_sim import FakeSim
from ..utils.array_utils import custom_len

from ..custom_exceptions import *
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

        super(AltHodModelFactory, self).__init__(input_model_blueprint, **kwargs)

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()
        self.model_blueprint = self._input_model_blueprint

        # Build the composite model dictionary, 
        # whose keys are parameters of our model
        self._set_init_param_dict()

        # Build up and bind several lists from the component models
        self._build_composite_lists()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()
        self._set_calling_sequence(**kwargs)
        self._test_blueprint_consistency()

        self.mock_factory = mock_factories.HodMockFactory


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

            gal_type_blueprint = self.model_blueprint[gal_type]

            feature_generator = (feature_name for feature_name in gal_type_blueprint
                if feature_name is not 'mock_factory')

            for feature_name in feature_generator:
                component_model_instance = gal_type_blueprint[feature_name]

                methods_to_inherit = list(set(
                    component_model_instance._methods_to_inherit))
                for methodname in methods_to_inherit:
                    new_method_name = methodname + '_' + gal_type
                    new_method_behavior = self._update_param_dict_decorator(
                        component_model_instance, methodname)
                    setattr(self, new_method_name, new_method_behavior)

                attrs_to_inherit = list(set(
                    component_model_instance._attrs_to_inherit))
                for attrname in attrs_to_inherit:
                    new_attr_name = attrname + '_' + gal_type
                    attr = getattr(component_model_instance, attrname)
                    setattr(self, new_attr_name, attr)

            # Repeatedly overwrite self.threshold 
            # This is harmless provided that all gal_types are ensured to have the same threshold, 
            # which is guaranteed by the _test_blueprint_consistency method
            self.threshold = getattr(self, 'threshold_' + gal_type)


    def _update_param_dict_decorator(self, component_model, func_name):
        """ Decorator used to propagate any possible changes 
        in the composite model param_dict 
        down to the appropriate component model param_dict. 
        """

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
        In MCMC applications, the items of ``param_dict`` define the possible 
        parameter set explored by the likelihood engine. 
        Changing the values of the parameters in ``param_dict`` 
        will propagate to the behavior of the component models 
        when the relevant methods are called. 
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

                if not hasattr(model_instance, 'param_dict'):
                    model_instance.param_dict = {}
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
        dtype_list = []
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

                # Column dtypes to add to mock galaxy_table
                if hasattr(component_model, '_galprop_dtypes_to_allocate'):
                    dtype_list.append(component_model._galprop_dtypes_to_allocate)

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

                # Ensure that all methods in the calling sequence are inherited
                try:
                    mock_making_methods = component_model._mock_generation_calling_sequence
                except AttributeError:
                    mock_making_methods = []
                try:
                    inherited_methods = component_model._methods_to_inherit
                except AttributeError:
                    inherited_methods = []
                    component_model._methods_to_inherit = []

                missing_methods = set(mock_making_methods) - set(inherited_methods).intersection(set(mock_making_methods))
                for methodname in missing_methods:
                    component_model._methods_to_inherit.append(methodname)

                if not hasattr(component_model, '_attrs_to_inherit'):
                    component_model._attrs_to_inherit = []


        self._haloprop_list = list(set(haloprop_list))
        self.prof_param_keys = list(set(prof_param_keys))
        self.publications = list(set(pub_list))
        self.new_haloprop_func_dict = new_haloprop_func_dict
        self._galprop_dtypes_to_allocate = model_helpers.create_composite_dtype(dtype_list)

    def _set_calling_sequence(self, **kwargs):
        """
        """
        if 'mock_generation_calling_sequence' in kwargs:
            self._mock_generation_calling_sequence = kwargs['mock_generation_calling_sequence']
            return 

        feature_keys = self.model_blueprint[self.model_blueprint.keys()[0]].keys()
        feature_keys.remove('occupation')
        feature_keys.remove('profile')
        feature_keys.insert(0, 'occupation')
        feature_keys.append('occupation')

        missing_calling_sequence_msg = ("\nComponent models typically have a list attribute called "
            "_mock_generation_calling_sequence.\nThis list determines the methods that are called "
            "by the mock factory, and the order in which they are called.\n"
            "The ``%s`` component of the gal_type = ``%s`` population has no such method.\n"
            "Only ignore this warning if you are sure this is not an error.\n")

        self._mock_generation_calling_sequence = []
        for gal_type in self.gal_types:
            for feature_key in feature_keys:
                component_model = self.model_blueprint[gal_type][feature_key]
                if hasattr(component_model, '_mock_generation_calling_sequence'):
                    component_methods = [name + '_' + gal_type for name in component_model._mock_generation_calling_sequence]
                    self._mock_generation_calling_sequence.extend(component_methods)
                else:
                    warn(missing_calling_sequence_msg % (feature_key, gal_type))


    def _test_blueprint_consistency(self):
        """
        Impose the following requirements on the blueprint: 

            * All occupation components have the same threshold. 

            * Each element in _mock_generation_calling_sequence is included in _methods_to_inherit
        """
        threshold_list = [getattr(self, 'threshold_' + gal_type) for gal_type in self.gal_types]
        if len(threshold_list) > 1:
            d = np.diff(threshold_list)
            if np.any(d != 0):
                threshold_msg = ''
                for gal_type in self.gal_types:
                    threshold_msg += '\n' + gal_type + ' threshold = ' + str(getattr(self, 'threshold_' + gal_type))
                msg = ("Inconsistency in the threshold of the component occupation models:\n" + threshold_msg + "\n")
                raise HalotoolsError(msg)

        missing_method_msg1 = ("\nAll component models have a ``_mock_generation_calling_sequence`` attribute,\n"
            "which is a list of method names that are called by the ``populate_mock`` method of the mock factory.\n"
            "All component models also have a ``_methods_to_inherit`` attribute, \n"
            "which determines which methods of the component model are inherited by the composite model.\n"
            "The former must be a subset of the latter. However, for ``gal_type`` = %s,\n"
            "the following method was not inherited:\n%s")
        for gal_type in self.gal_types:
            for component_model in self.model_blueprint[gal_type].values():
                mock_generation_methods = set(component_model._mock_generation_calling_sequence)
                inherited_methods = set(component_model._methods_to_inherit)
                overlap = mock_generation_methods.intersection(inherited_methods)
                missing_methods = mock_generation_methods - overlap
                if missing_methods != set():
                    some_missing_method = list(missing_methods)[0]
                    raise HalotoolsError(missing_method_msg1 % (gal_type, some_missing_method))

        missing_method_msg2 = ("\nAll component models have a ``_mock_generation_calling_sequence`` attribute,\n"
            "which is a list of method names that are called by the ``populate_mock`` method of the mock factory.\n"
            "The AltHodModelFactory builds a composite ``_mock_generation_calling_sequence`` from each of these lists.\n"
            "However, the following method does not appear to have been created during this process:\n%s\n"
            "This is likely a bug in Halotools - please raise an Issue on https://github.com/astropy/halotools\n")
        for method in self._mock_generation_calling_sequence:
            if not hasattr(self, method):
                raise HalotoolsError(missing_method_msg2)









##########################################











