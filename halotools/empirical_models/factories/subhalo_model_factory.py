# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__all__ = ['SubhaloModelFactory']
__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from warnings import warn 
import collections 

from ..factories import ModelFactory, SubhaloMockFactory

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
    and generates a composite model object. The returned object can be used directly to 
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
            Function should take a length-k Astropy table as a single positional argument, 
            and return a length-k numpy boolean array that will be 
            treated as a mask over the rows of the table. If not None, 
            the mask defined by ``galaxy_selection_func`` will be applied to the 
            ``galaxy_table`` after the table is generated by the `populate_mock` method. 
            Default is None.  

        halo_selection_func : function object, optional   
            Function object used to place a cut on the input ``halo_table``. 
            If the ``halo_selection_func`` keyword argument is passed, 
            the input to the function must be a single positional argument storing a 
            length-N structured numpy array or Astropy table; 
            the function output must be a length-N boolean array that will be used as a mask. 
            Halos that are masked will be entirely neglected during mock population.
        """

        super(SubhaloModelFactory, self).__init__(input_model_blueprint, **kwargs)
        self.model_blueprint = copy(self._input_model_blueprint)
        
        # Build up and bind several lists from the component models
        self._build_composite_attrs(**kwargs)

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()

        self.mock_factory = SubhaloMockFactory

    def _build_composite_attrs(self, **kwargs):
        """ A composite model has several bookkeeping devices that are built up from 
        the components: ``_haloprop_list``, ``publications``, and ``new_haloprop_func_dict``. 
        """

        self._feature_list = self.model_blueprint.keys()

        self._build_haloprop_list()
        self._build_publication_list()
        self._build_new_haloprop_func_dict()
        self._build_dtype_list()
        self._set_warning_suppressions()
        self._set_model_redshift()
        self._set_inherited_methods()
        self._set_init_param_dict()

    def _set_inherited_methods(self):
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

        After calling the _set_inherited_methods method, it will be therefore be entirely safe to 
        run a for loop over each component model's `_methods_to_inherit` and `_attrs_to_inherit`, 
        even if these lists were forgotten or irrelevant to that particular component. 
        """

        _method_repetition_check = []
        _attrs_repetition_check = []

        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

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

            _method_repetition_check.extend(component_model._methods_to_inherit)

            if not hasattr(component_model, '_attrs_to_inherit'):
                component_model._attrs_to_inherit = []

            _attrs_repetition_check.extend(component_model._attrs_to_inherit)


        # Check that we do not have any method names to inherit that appear 
        # in more than one component model
        repeated_method_msg = ("\n The method name ``%s`` appears "
            "in more than one component model.\n You should rename this method in one of your "
            "component models to disambiguate.\n")
        repeated_method_list = ([methodname for methodname, count in 
            collections.Counter(_method_repetition_check).items() if count > 1]
            )
        if repeated_method_list != []:
            example_repeated_methodname = repeated_method_list[0]
            raise HalotoolsError(repeated_method_msg % example_repeated_methodname)

        # Check that we do not have any attributes to inherit that appear 
        # in more than one component model
        repeated_attr_msg = ("\n The attribute name ``%s`` appears "
            "in more than one component model.\n "
            "Only ignore this message if you are confident "
            "that this will not result in unintended behavior\n")
        repeated_attr_list = ([attr for attr, count in 
            collections.Counter(_attrs_repetition_check).items() if count > 1]
            )
        if repeated_attr_list != []:
            example_repeated_attr = repeated_attr_list[0]
            warn(repeated_attr_msg % example_repeated_attr)

    def _set_primary_behaviors(self, **kwargs):
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

        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

            try:
                component_model_galprop_dtype = component_model._galprop_dtypes_to_allocate
            except AttributeError:
                component_model_galprop_dtype = np.dtype([])

            methods_to_inherit = list(set(
                component_model._methods_to_inherit))

            for methodname in methods_to_inherit:
                new_method_name = methodname
                new_method_behavior = self._update_param_dict_decorator(
                    component_model, methodname)
                setattr(self, new_method_name, new_method_behavior)
                setattr(getattr(self, new_method_name), 
                    '_galprop_dtypes_to_allocate', component_model_galprop_dtype)

            attrs_to_inherit = list(set(
                component_model._attrs_to_inherit))
            for attrname in attrs_to_inherit:
                new_attr_name = attrname
                attr = getattr(component_model, attrname)
                setattr(self, new_attr_name, attr)

        self._set_calling_sequence(**kwargs)

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

    def _set_calling_sequence(self, **kwargs):
        """
        """
        self._mock_generation_calling_sequence = []

        missing_calling_sequence_msg = ("\nComponent models typically have a list attribute called "
            "_mock_generation_calling_sequence.\nThis list determines the methods that are called "
            "by the mock factory, and the order in which they are called.\n"
            "The ``%s`` component model has no such method.\n"
            "Only ignore this warning if you are sure this is not an error.\n")

        repeated_calling_sequence_element_msg = ("\n The method name ``%s`` that appears "
            "in the calling sequence of the \n``%s`` component model also appears in the "
            "calling sequence of another model.\nYou should rename this method in one of your "
            "component models to disambiguate.\n")

        ###############
        # If provided, retrieve the input list defining the calling sequence.
        # Otherwise, it will be assumed that specifying the calling sequence is not necessary 
        # and an effectively random sequence will be chosen
        try:
            feature_sequence = kwargs['mock_generation_calling_sequence']
        except KeyError:
            feature_sequence = self.model_blueprint.keys()

        ###############
        # Loop over feature_sequence and successively append each component model's
        # calling sequence to the composite model calling sequence
        for feature in feature_sequence:
            component_model = self.model_blueprint[feature]
            if hasattr(component_model, '_mock_generation_calling_sequence'):

                component_method_list = (
                    [name for name in component_model._mock_generation_calling_sequence]
                    )

                # test to make sure we have no repeated method names
                intersection = set(component_method_list) & set(self._mock_generation_calling_sequence)
                if intersection != set():
                    methodname = list(intersection)[0]
                    t = (methodname, component_model.__class__.__name__)
                    raise HalotoolsError(repeated_calling_sequence_element_msg % t)

                self._mock_generation_calling_sequence.extend(component_method_list)
            else:
                warn(missing_calling_sequence_msg % component_model.__class__.__name__)

    def _set_model_redshift(self):
        """ 
        """
        msg = ("Inconsistency between the redshifts of the component models:\n"
            "    For component model 1 = ``%s``, the model has redshift = %.2f.\n"
            "    For component model 2 = ``%s``, the model has redshift = %.2f.\n")

        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

            if hasattr(component_model, 'redshift'):
                redshift = component_model.redshift 
                try:
                    if redshift != existing_redshift:
                        t = (component_model.__class__.__name__, redshift, 
                            last_component.__class__.__name__, existing_redshift)
                        raise HalotoolsError(msg % t)
                except NameError:
                    existing_redshift = redshift 

            last_component = component_model

        self.redshift = redshift

    def _build_haloprop_list(self):
        """
        """
        haloprop_list = []
        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

            if hasattr(component_model, 'prim_haloprop_key'):
                haloprop_list.append(component_model.prim_haloprop_key)
            if hasattr(component_model, 'sec_haloprop_key'):
                haloprop_list.append(component_model.sec_haloprop_key)

        self._haloprop_list = list(set(haloprop_list))

    def _build_publication_list(self):
        """
        """
        pub_list = []
        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

            try:
                pubs = component_model.publications 
                if type(pubs) in [str, unicode]:
                    pub_list.append(pubs)
                elif type(pubs) is list:
                    pub_list.extend(pubs)
                else:
                    clname = component_model.__class__.__name__
                    msg = ("The ``publications`` attribute of the " + clname + " feature\n"
                        "must be a string or list of strings")
                    raise HalotoolsError(msg)
            except AttributeError:
                pass

        self.publications = list(set(pub_list))

    def _build_new_haloprop_func_dict(self):
        """
        """
        new_haloprop_func_dict = {}
        # Loop over all component features in the composite model
        for feature in self._feature_list:
            component_model = self.model_blueprint[feature]

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
                    clname = component_model.__class__.__name__
                    msg = ("The composite model received multiple "
                        "component models \nwith a new_haloprop_func_dict that use "
                        "the %s key. \nIgnoring the one that appears in the %s feature")
                    warn(msg % (example_repeated_element, clname))

        self.new_haloprop_func_dict = new_haloprop_func_dict

    def _set_warning_suppressions(self):
        """
        """
        self._suppress_repeated_param_warning = False
        # Loop over all component features in the composite model
        for feature_key in self._feature_list:
            component_model = self.model_blueprint[feature_key]
            if hasattr(component_model, '_suppress_repeated_param_warning'):
                self._suppress_repeated_param_warning += component_model._suppress_repeated_param_warning

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

        # Loop over all component features in the composite model
        for feature_key in self._feature_list:
            component_model = self.model_blueprint[feature_key]

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

    def _build_dtype_list(self):
        """
        """
        dtype_list = []
        # Loop over all component features in the composite model
        for feature_key in self._feature_list:
            component_model = self.model_blueprint[feature_key]

            # Column dtypes to add to mock galaxy_table
            if hasattr(component_model, '_galprop_dtypes_to_allocate'):
                dtype_list.append(component_model._galprop_dtypes_to_allocate)

        self._galprop_dtypes_to_allocate = model_helpers.create_composite_dtype(dtype_list)

    def restore_init_param_dict(self):
        """ Reset all values of the current ``param_dict`` to the values 
        the class was instantiated with. 

        Primary behaviors are reset as well, as this is how the 
        inherited behaviors get bound to the values in ``param_dict``. 
        """
        self.param_dict = self._init_param_dict
        self._set_primary_behaviors()
