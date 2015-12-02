# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__all__ = ['HodModelFactory']
__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from functools import partial
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from warnings import warn 
import collections 

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



class HodModelFactory(ModelFactory):
    """ Class used to build HOD-style models of the galaxy-halo connection. 

    The arguments passed to the `HodModelFactory` constructor determine 
    the features of the model that are returned by the factory. This works as follows. 
    A sequence of ``model_features`` can be passed, each of which 
    are instances of component models. The factory composes these independently-defined 
    components into a composite model. The returned instance can be used directly to 
    populate a simulation with a Monte Carlo realization of the model, as shown in the examples. 

    Most behavior is derived from external classes bound up in the input ``model_dictionary``. 
    So the purpose of `HodModelFactory` is mostly to compose these external 
    behaviors together into a composite model. 
    The aim is to provide a standardized model object 
    that interfaces consistently with the rest of the package, 
    regardless of the features of the model. 
    
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        *model_features : sequence of keyword arguments, optional 
            The standard way to call the `HodModelFactory` is 
            with a sequence of keyword arguments providing the set of 
            features that you want to build your composite model with. 
            Each keyword you use will be simultaneously interpreted as 
            the name of the feature and the name of the galaxy population 
            with that feature; the value bound to each keyword 
            must be an instance of a component model governing 
            the behavior of that feature. See the examples section below. 

        model_feature_calling_sequence : list, optional
            Determines the order in which your component features  
            will be called during mock population. 

            Some component models may have explicit dependence upon 
            the value of some other galaxy model property. 
            In such a case, you must pass a ``model_feature_calling_sequence`` list, 
            ordered in the desired calling sequence. 
            A classic example is if the stellar mass of a central galaxy has explicit 
            dependence on whether or not the central is active or quiescent. 
            In such a case, an example ``model_feature_calling_sequence`` could be 
            model_feature_calling_sequence = ['centrals_quiescent', 'centrals_occupation', ...]

            Default behavior is to assume that no model feature  
            has explicit dependence upon any other, in which case the component 
            models appearing in the ``model_features`` keyword arguments 
            will be called in random order. 

        gal_type_list : list, optional 
            List of strings providing the names of the galaxy types in the 
            composite model. This is only necessary to provide if you have 
            a gal_type in your model that is neither ``centrals`` nor ``satellites``. 

            For example, if you have entirely separate models for ``red_satellites`` and 
            ``blue_satellites``, then your ``gal_type_list`` might be, 
            gal_type_list = ['centrals', 'red_satellites', 'blue_satellites']. 
            Another possible example would be 
            gal_type_list = ['centrals', 'satellites', 'orphans']. 

        halo_selection_func : function object, optional   
            Function object used to place a cut on the input ``table``. 
            If the ``halo_selection_func`` keyword argument is passed, 
            the input to the function must be a single positional argument storing a 
            length-N structured numpy array or Astropy table; 
            the function output must be a length-N boolean array that will be used as a mask. 
            Halos that are masked will be entirely neglected during mock population.

        Examples 
        ---------
        >>> from halotools.empirical_models import leauthaud11_model_dictionary
        >>> model_dictionary = leauthaud11_model_dictionary(redshift = 2, threshold = 10.5)
        >>> model_instance = HodModelFactory(**model_dictionary)

        All model instance have a `populate_mock` method that allows you to generate Monte Carlo 
        realizations of galaxy populations based on the underlying analytical functions: 

        >>> model_instance.populate_mock(simname = 'bolshoi', redshift = 2) # doctest: +SKIP

        There also convenience functions for estimating the clustering signal predicted by the model. 
        For example, the following method repeatedly populates the Bolshoi simulation with 
        galaxies, computes the 3-d galaxy clustering signal of each mock, computes the median 
        clustering signal in each bin, and returns the result:

        >>> r, xi = model_instance.compute_average_galaxy_clustering(num_iterations = 5, redshift = 2) # doctest: +SKIP

        """

        input_model_dictionary, supplementary_kwargs = self._parse_constructor_kwargs(
            **kwargs)

        super(HodModelFactory, self).__init__(input_model_dictionary, **supplementary_kwargs)

        self.mock_factory = HodMockFactory
        self.model_factory = HodModelFactory

        self._model_feature_calling_sequence = (
            self.build_model_feature_calling_sequence(supplementary_kwargs))

        self.model_dictionary = collections.OrderedDict()
        for key in self._model_feature_calling_sequence:
            self.model_dictionary[key] = copy(self._input_model_dictionary[key])

        # Build up and bind several lists from the component models
        self.set_gal_types()
        self.build_prim_sec_haloprop_list()
        self.build_prof_param_keys()
        self.build_publication_list()
        self.build_dtype_list()
        self.build_new_haloprop_func_dict()
        self.set_warning_suppressions()
        self.set_inherited_methods()
        self.set_model_redshift()
        self.build_init_param_dict()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self.set_primary_behaviors()
        self.set_calling_sequence()
        self._test_dictionary_consistency()

        ############################################################

    def _parse_constructor_kwargs(self, **kwargs):
        """
        """
        input_model_dictionary = copy(kwargs)

        ### First parse the supplementary keyword arguments, 
        # such as 'model_feature_calling_sequence', 
        ### from the keywords that are bound to component model instances, 
        # such as 'centrals_occupation'

        possible_supplementary_kwargs = (
            'halo_selection_func', 
            'model_feature_calling_sequence', 
            'gal_type_list'
            )

        supplementary_kwargs = {}
        for key in possible_supplementary_kwargs:
            try:
                supplementary_kwargs[key] = copy(input_model_dictionary[key])
                del input_model_dictionary[key]
            except KeyError:
                pass

        if 'gal_type_list' not in supplementary_kwargs:
            supplementary_kwargs['gal_type_list'] = None

        if 'model_feature_calling_sequence' not in supplementary_kwargs:
            supplementary_kwargs['model_feature_calling_sequence'] = None

        return input_model_dictionary, supplementary_kwargs


    def build_model_feature_calling_sequence(self, supplementary_kwargs):
        """
        """

        ########################
        ### Require that all elements of the input model_feature_calling_sequence 
        ### were also keyword arguments to the __init__ constructor 
        try:
            model_feature_calling_sequence = list(supplementary_kwargs['model_feature_calling_sequence'])
            for model_feature in model_feature_calling_sequence:
                try:
                    assert model_feature in self._input_model_dictionary.keys()
                except AssertionError:
                    msg = ("\nYour input ``model_feature_calling_sequence`` has a ``%s`` element\n"
                    "that does not appear in the keyword arguments you passed to the HodModelFactory.\n"
                    "For every element of the input ``model_feature_calling_sequence``, there must be a corresponding \n"
                    "keyword argument to which a component model instance is bound.\n")
                    raise HalotoolsError(msg % model_feature)
        except TypeError:
            # The supplementary_kwargs['model_feature_calling_sequence'] was None, triggering a TypeError, 
            # so we will use the default calling sequence instead
            # The default sequence will be to first use the centrals_occupation (if relevant), 
            # then any possible additional occupation features, then any possible remaining features
            model_feature_calling_sequence = []

            occupation_keys = [key for key in self._input_model_dictionary if 'occupation' in key]
            centrals_occupation_keys = [key for key in occupation_keys if 'central' in key]
            remaining_occupation_keys = [key for key in occupation_keys if key not in centrals_occupation_keys]

            model_feature_calling_sequence.extend(centrals_occupation_keys)
            model_feature_calling_sequence.extend(remaining_occupation_keys)

            remaining_model_dictionary_keys = (
                [key for key in self._input_model_dictionary if key not in model_feature_calling_sequence]
                )
            model_feature_calling_sequence.extend(remaining_model_dictionary_keys)

        ########################

        ########################
        ### Now conversely require that all remaining __init__ constructor keyword arguments 
        ### appear in the model_feature_calling_sequence
        for constructor_kwarg in self._input_model_dictionary:
            try:
                assert constructor_kwarg in model_feature_calling_sequence
            except AssertionError:
                msg = ("\nYou passed ``%s`` as a keyword argument to the HodModelFactory constructor.\n"
                    "This keyword argument does not appear in your input ``model_feature_calling_sequence``\n"
                    "and is otherwise not recognized.\n")
                raise HalotoolsError(msg % constructor_kwarg)
        ########################

        gal_type_list = supplementary_kwargs['gal_type_list']

        self._test_model_feature_calling_sequence_consistency(
            model_feature_calling_sequence, gal_type_list)

        return model_feature_calling_sequence


    def _test_model_feature_calling_sequence_consistency(self, 
        model_feature_calling_sequence, gal_type_list):
        """
        """        
        for model_feature_calling_sequence_element in model_feature_calling_sequence:

            try:
                component_model = self._input_model_dictionary[model_feature_calling_sequence_element]
            except KeyError:
                msg = ("\nYour input ``model_feature_calling_sequence`` has a ``%s`` element\n"
                    "that does not appear in the keyword arguments passed to \n"
                    "the constructor of the HodModelFactory.\n")
                raise HalotoolsError(msg % model_feature_calling_sequence_element)

            component_model_class_name = component_model.__class__.__name__

            gal_type, feature_name = self._infer_gal_type_and_feature_name(
                model_feature_calling_sequence_element, gal_type_list)

            try:
                component_model_gal_type = component_model.gal_type
            except AttributeError:
                self._input_model_dictionary[model_feature_calling_sequence_element].gal_type = gal_type
                component_model_gal_type = gal_type

            try:
                component_model_feature_name = component_model.feature_name
            except AttributeError:
                self._input_model_dictionary[model_feature_calling_sequence_element].feature_name = feature_name
                component_model_feature_name = feature_name

            try:
                assert gal_type == component_model_gal_type
            except AssertionError:
                msg = ("\nThe ``%s`` component model instance has ``gal_type`` = %s.\n"
                    "However, you used a keyword argument = ``%s`` when passing this component model \n"
                    "to the constructor of the HodModelFactory, \nfrom which it was inferred that your intended"
                    "``gal_type`` = %s, which is inconsistent.\nIf this inferred ``gal_type`` seems incorrect,\n"
                    "please raise an Issue on https://github.com/astropy/halotools.\n"
                    "Otherwise, either change the ``%s`` keyword argument to conform to the Halotools convention \n"
                    "to use keyword arguments that are composed of a ``gal_type`` and ``feature_name`` substring,\n"
                    "separated by a '_', in that order.\n")
                raise HalotoolsError(msg % 
                    (component_model_class_name, component_model_gal_type, model_feature_calling_sequence_element, 
                        gal_type, model_feature_calling_sequence_element))

            try:
                assert feature_name == component_model_feature_name
            except AssertionError:
                msg = ("\nThe ``%s`` component model instance has ``feature_name`` = %s.\n"
                    "However, you used a keyword argument = ``%s`` when passing this component model \n"
                    "to the constructor of the HodModelFactory, \nfrom which it was inferred that your intended"
                    "``feature_name`` = %s, which is inconsistent.\nIf this inferred ``feature_name`` seems incorrect,\n"
                    "please raise an Issue on https://github.com/astropy/halotools.\n"
                    "Otherwise, either change the ``%s`` keyword argument to conform to the Halotools convention \n"
                    "to use keyword arguments that are composed of a ``gal_type`` and ``feature_name`` substring,\n"
                    "separated by a '_', in that order.\n")
                raise HalotoolsError(msg % 
                    (component_model_class_name, component_model_feature_name, model_feature_calling_sequence_element, 
                        feature_name, model_feature_calling_sequence_element))


    def _infer_gal_type_and_feature_name(self, model_dictionary_key, gal_type_list, 
        known_gal_type = None, known_feature_name = None):
        
        processed_key = model_dictionary_key.lower()
        
        if known_gal_type is not None:
            gal_type = known_gal_type
            
            # Ensure that the gal_type appears first in the string
            if processed_key[0:len(gal_type)] != gal_type:
                msg = ("\nThe first substring of each key of the ``model_dictionary`` \n"
                    "must be the ``gal_type`` substring. So the first substring of the ``%s`` key \n"
                    "should be %s")
                raise HalotoolsError(msg % (model_dictionary_key, gal_type))
                    
            # Remove the gal_type substring
            processed_key = processed_key.replace(gal_type, '')
            
            # Ensure that the gal_type and feature_name were separated by a '_'
            if processed_key[0] != '_':
                msg = ("\nThe model_dictionary key ``%s`` must be comprised of \n"
                    "the ``gal_type`` and ``feature_name`` substrings, separated by a '_', in that order.\n")
                raise HalotoolsError(msg % model_dictionary_key)
            else:
                processed_key = processed_key[1:]
                feature_name = processed_key
            return gal_type, feature_name
        
        elif known_feature_name is not None:
            feature_name = known_feature_name
            
            # Ensure that the feature_name appears last in the string
            feature_name_first_idx = processed_key.find(feature_name)
            if processed_key[feature_name_first_idx:] != feature_name:
                msg = ("\nThe second substring of each key of the ``model_dictionary`` \n"
                    "must be the ``feature_name`` substring. So the second substring of the ``%s`` key \n"
                    "should be %s")
                raise HalotoolsError(msg % (model_dictionary_key, feature_name))
                
            # Remove the feature_name substring
            processed_key = processed_key.replace(feature_name, '')
         
            # Ensure that the gal_type and feature_name were separated by a '_'
            if processed_key[-1] != '_':
                msg = ("\nThe model_dictionary key ``%s`` must be comprised of \n"
                    "the ``gal_type`` and ``feature_name`` substrings, separated by a '_', in that order.\n")
                raise HalotoolsError(msg % model_dictionary_key)
            else:
                processed_key = processed_key[:-1]
                gal_type = processed_key
            return gal_type, feature_name
        else:
            if gal_type_list is not None:
                gal_type_guess_list = gal_type_list 
            else:
                gal_type_guess_list = ('centrals', 'satellites')

            for gal_type_guess in gal_type_guess_list:                
                if gal_type_guess in processed_key:
                    known_gal_type = gal_type_guess
                    gal_type, feature_name = self._infer_gal_type_and_feature_name(
                        processed_key, gal_type_guess_list, known_gal_type = known_gal_type)
                    return gal_type, feature_name
            msg = ("\nThe ``_infer_gal_type_and_feature_name`` method was unable to identify\n"
                "the name of your galaxy population from the ``%s`` key of the model_dictionary.\n"
                "If you are modeling a population whose name is neither ``centrals`` nor ``satellites``,\n"
                "then you must provide a ``gal_type_list`` keyword argument to \n"
                "the constructor of the HodModelFactory.\n")
            raise HalotoolsError(msg % model_dictionary_key)


    def set_gal_types(self):
        """ Private method binding the ``gal_types`` list attribute. 
        If there are both centrals and satellites, method ensures that centrals 
        will always be built first, out of consideration for satellite 
        model components with explicit dependence on the central population. 
        """
        _gal_type_list = []
        for component_model in self.model_dictionary.values():
            _gal_type_list.append(component_model.gal_type)
        self.gal_types = set(list(_gal_type_list))


    def set_primary_behaviors(self):
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
        `set_primary_behaviors` just creates a symbolic link to those external behaviors. 
        """

        for component_model in self.model_dictionary.values():
            gal_type = component_model.gal_type
            feature_name = component_model.feature_name

            try:
                component_model_galprop_dtype = component_model._galprop_dtypes_to_allocate
            except AttributeError:
                component_model_galprop_dtype = np.dtype([])

            methods_to_inherit = list(set(
                component_model._methods_to_inherit))

            for methodname in methods_to_inherit:
                new_method_name = methodname + '_' + gal_type
                new_method_behavior = self.update_param_dict_decorator(
                    component_model, methodname)
                setattr(self, new_method_name, new_method_behavior)
                setattr(getattr(self, new_method_name), 
                    '_galprop_dtypes_to_allocate', component_model_galprop_dtype)
                setattr(getattr(self, new_method_name), 'gal_type', gal_type)
                setattr(getattr(self, new_method_name), 'feature_name', feature_name)

            attrs_to_inherit = list(set(
                component_model._attrs_to_inherit))
            for attrname in attrs_to_inherit:
                new_attr_name = attrname + '_' + gal_type
                attr = getattr(component_model, attrname)
                setattr(self, new_attr_name, attr)

            # Repeatedly overwrite self.threshold 
            # This is harmless provided that all gal_types are ensured to have the same threshold, 
            # which is guaranteed by the _test_dictionary_consistency method
            if hasattr(component_model, 'threshold'):
                setattr(self, 'threshold_' + gal_type, component_model.threshold)
                self.threshold = getattr(self, 'threshold_' + gal_type)


    def update_param_dict_decorator(self, component_model, func_name):
        """ Decorator used to propagate any possible changes in the composite model param_dict 
        down to the appropriate component model param_dict. 

        Parameters 
        -----------
        component_model : obj 
            Instance of the component model in which the behavior of the function is defined. 

        func_name : string 
            Name of the method in the component model whose behavior is being decorated. 

        Returns 
        --------
        decorated_func : function 
            Function object whose behavior is identical 
            to the behavior of the function in the component model, 
            except that the component model param_dict is first updated with any 
            possible changes to corresponding parameters in the composite model param_dict.

        See also 
        --------
        :ref:`update_param_dict_decorator_mechanism`

        :ref:`param_dict_mechanism`
        """
        return ModelFactory.update_param_dict_decorator(self, component_model, func_name)

    def build_lookup_tables(self):
        """ Method to compute and load lookup tables for each of 
        the phase space component models. 
        """

        for component_model in self.model_dictionary.values():
            if hasattr(component_model, 'build_lookup_tables'):
                component_model.build_lookup_tables()

    def build_init_param_dict(self):
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

        try:
            suppress_warning = self._suppress_repeated_param_warning
        except AttributeError:
            suppress_warning = False
        msg = ("\n\nThe param_dict key %s appears in more than one component model.\n"
            "This is permissible, but if you are seeing this message you should be sure you "
            "understand it.\nIn particular, double-check that this parameter does not have "
            "conflicting meanings across components.\n"
            "\nIf you do not wish to see this message every time you instantiate, \n"
            "simply attach a _suppress_repeated_param_warning attribute \n"
            "to any of your component models and set this variable to ``True``.\n")

        for component_model in self.model_dictionary.values():

            if not hasattr(component_model, 'param_dict'):
                component_model.param_dict = {}
            intersection = set(self.param_dict) & set(component_model.param_dict)
            if intersection != set():
                for key in intersection:
                    if suppress_warning is False:
                        warn(msg % key)

            for key, value in component_model.param_dict.iteritems():
                self.param_dict[key] = value

        self._init_param_dict = copy(self.param_dict)

    def restore_init_param_dict(self):
        """ Reset all values of the current ``param_dict`` to the values 
        the class was instantiated with. 

        Primary behaviors are reset as well, as this is how the 
        inherited behaviors get bound to the values in ``param_dict``. 
        """
        self.param_dict = self._init_param_dict
        self.set_primary_behaviors()
        self.set_calling_sequence()

    def set_model_redshift(self):
        """ 
        """
        msg = ("Inconsistency between the redshifts of the component models:\n"
            "    For gal_type = ``%s``, the %s model has redshift = %.2f.\n"
            "    For gal_type = ``%s``, the %s model has redshift = %.2f.\n")


        for component_model in self.model_dictionary.values():
            gal_type = component_model.gal_type

            if hasattr(component_model, 'redshift'):
                redshift = component_model.redshift 
                try:
                    if redshift != existing_redshift:
                        t = (gal_type, component_model.__class__.__name__, redshift, 
                            last_gal_type, last_component.__class__.__name__, existing_redshift)
                        raise HalotoolsError(msg % t)
                except NameError:
                    existing_redshift = redshift 

            last_component = component_model
            last_gal_type = gal_type

        try:
            self.redshift = redshift
        except NameError:
            self.redshift = sim_defaults.default_redshift


    def build_prim_sec_haloprop_list(self):
        """
        """
        haloprop_list = []
        for component_model in self.model_dictionary.values():

            if hasattr(component_model, 'prim_haloprop_key'):
                haloprop_list.append(component_model.prim_haloprop_key)
            if hasattr(component_model, 'sec_haloprop_key'):
                haloprop_list.append(component_model.sec_haloprop_key)

        self._haloprop_list = list(set(haloprop_list))

    def build_prof_param_keys(self):
        """
        """
        prof_param_keys = []

        for component_model in self.model_dictionary.values():
            if hasattr(component_model, 'prof_param_keys'):
                prof_param_keys.extend(component_model.prof_param_keys)

        self.prof_param_keys = list(set(prof_param_keys))

    def build_publication_list(self):
        """
        """
        pub_list = []
        for component_model in self.model_dictionary.values():

            if hasattr(component_model, 'publications'):
                pub_list.extend(component_model.publications)

        self.publications = list(set(pub_list))

    def build_dtype_list(self):
        """
        """
        dtype_list = []
        for component_model in self.model_dictionary.values():

            # Column dtypes to add to mock galaxy_table
            if hasattr(component_model, '_galprop_dtypes_to_allocate'):
                dtype_list.append(component_model._galprop_dtypes_to_allocate)

        self._galprop_dtypes_to_allocate = model_helpers.create_composite_dtype(dtype_list)

    def build_new_haloprop_func_dict(self):
        """
        """
        new_haloprop_func_dict = {}

        for component_model in self.model_dictionary.values():
            feature_name, gal_type = component_model.feature_name, component_model.gal_type

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
                    warn(msg % (example_repeated_element, feature_name, gal_type))

        self.new_haloprop_func_dict = new_haloprop_func_dict

    def set_warning_suppressions(self):
        """
        """
        self._suppress_repeated_param_warning = False

        for component_model in self.model_dictionary.values():

            if hasattr(component_model, '_suppress_repeated_param_warning'):
                self._suppress_repeated_param_warning += component_model._suppress_repeated_param_warning

    def set_inherited_methods(self):
        """ Each component model *should* have a `_mock_generation_calling_sequence` attribute 
        that provides the sequence of method names to call during mock population. Additionally, 
        each component *should* have a `_methods_to_inherit` attribute that determines 
        which methods will be inherited by the composite model. 
        The `_mock_generation_calling_sequence` list *should* be a subset of `_methods_to_inherit`. 
        If any of the above conditions fail, no exception will be raised during the construction 
        of the composite model. Instead, an empty list will be forcibly attached to each 
        component model for which these lists may have been missing. 
        Also, for each component model, if there are any elements of `_mock_generation_calling_sequence` 
        that were missing from `_methods_to_inherit`, all such elements will be forcibly added to 
        that component model's `_methods_to_inherit`.

        Finally, each component model *should* have an `_attrs_to_inherit` attribute that determines 
        which attributes will be inherited by the composite model. If any component models did not 
        implement the `_attrs_to_inherit`, an empty list is forcibly added to the component model. 

        After calling the set_inherited_methods method, it will be therefore be entirely safe to 
        run a for loop over each component model's `_methods_to_inherit` and `_attrs_to_inherit`, 
        even if these lists were forgotten or irrelevant to that particular component. 
        """

        for component_model in self.model_dictionary.values():

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


    def set_calling_sequence(self):
        """
        """
        # model_feature_calling_sequence
        self._mock_generation_calling_sequence = []

        missing_calling_sequence_msg = ("\nComponent models typically have a list attribute called "
            "_mock_generation_calling_sequence.\nThis list determines the methods that are called "
            "by the mock factory, and the order in which they are called.\n"
            "The ``%s`` component of the gal_type = ``%s`` population has no such method.\n"
            "Only ignore this warning if you are sure this is not an error.\n")

        for model_feature in self._model_feature_calling_sequence:
            component_model = self.model_dictionary[model_feature]

            if hasattr(component_model, '_mock_generation_calling_sequence'):
                component_method_list = (
                    [name + '_' + component_model.gal_type 
                    for name in component_model._mock_generation_calling_sequence]
                    )
                self._mock_generation_calling_sequence.extend(component_method_list)
            else:
                warn(missing_calling_sequence_msg % (component_model.feature_name, component_model.gal_type))


    def _test_dictionary_consistency(self):
        """
        Impose the following requirements on the dictionary: 

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
        for component_model in self.model_dictionary.values():

            mock_generation_methods = set(component_model._mock_generation_calling_sequence)
            inherited_methods = set(component_model._methods_to_inherit)
            overlap = mock_generation_methods.intersection(inherited_methods)
            missing_methods = mock_generation_methods - overlap
            if missing_methods != set():
                some_missing_method = list(missing_methods)[0]
                raise HalotoolsError(missing_method_msg1 % (component_model.gal_type, some_missing_method))

        missing_method_msg2 = ("\nAll component models have a ``_mock_generation_calling_sequence`` attribute,\n"
            "which is a list of method names that are called by the ``populate_mock`` method of the mock factory.\n"
            "The HodModelFactory builds a composite ``_mock_generation_calling_sequence`` from each of these lists.\n"
            "However, the following method does not appear to have been created during this process:\n%s\n"
            "This is likely a bug in Halotools - please raise an Issue on https://github.com/astropy/halotools\n")
        for method in self._mock_generation_calling_sequence:
            if not hasattr(self, method):
                raise HalotoolsError(missing_method_msg2)

##########################################

