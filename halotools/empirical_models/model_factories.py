# -*- coding: utf-8 -*-
"""
Module storing the various factories used to build galaxy-halo models. 
"""

__all__ = ['ModelFactory', 'SubhaloModelFactory', 'HodModelFactory']
__author__ = ['Andrew Hearin']

import numpy as np
from copy import copy
from functools import partial

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import occupation_helpers as occuhelp
from . import model_defaults
from . import mock_factories
from . import preloaded_hod_blueprints
from . import gal_prof_factory
from . import halo_prof_components

from ..sim_manager.read_nbody import ProcessedSnapshot
from ..sim_manager.generate_random_sim import FakeSim
from ..utils.array_utils import array_like_length as custom_len


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
                # or else the call to mock_factories below 
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

        """

        super(SubhaloModelFactory, self).__init__(input_model_blueprint, **kwargs)

        self.model_blueprint = self._interpret_input_model_blueprint()
        
        self._build_composite_lists(**kwargs)

        self._set_primary_behaviors()


    def _interpret_input_model_blueprint(self):

        model_blueprint = copy(self._input_model_blueprint)

        if 'mock_factory' not in model_blueprint.keys():
            model_blueprint['mock_factory'] = mock_factories.SubhaloMockFactory

        return model_blueprint

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

    def _build_composite_lists(self, **kwargs):
        """ A composite model has several bookkeeping devices that are built up from 
        the components: ``_haloprop_list``, ``publications``, and ``new_haloprop_func_dict``. 
        """

        unordered_galprop_list = [key for key in self.model_blueprint.keys() if key is not 'mock_factory']
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
                    new_haloprop_func_dict = (
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



class HodModelFactory(ModelFactory):
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
    
        * The following tutorial, :ref:`custom_hod_model_building_tutorial`, shows how you can build your own, customizing it based on the science you are interested in.  

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
            See :ref:`custom_hod_model_building_tutorial` for further details. 

        """

        super(HodModelFactory, self).__init__(input_model_blueprint, **kwargs)

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()
        self.model_blueprint = self.interpret_input_model_blueprint()

        # Build the composite model dictionary, 
        # whose keys are parameters of our model
        self._set_init_param_dict()

        # Build up and bind several lists from the component models
        self._build_composite_lists()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()

    def interpret_input_model_blueprint(self):
        """ Method to interpret the ``input_model_blueprint`` 
        passed to the constructor into ``self.model_blueprint``: 
        the set of instructions that are actually used 
        by `HodModelFactory` to create the model. 

        Notes 
        ----- 
        In order for `HodModelFactory` to build a composite model object, 
        each galaxy's ``profile`` key of the ``model_blueprint`` 
        must be an instance of the 
        `~halotools.empirical_models.GalProfFactory` class. 
        However, if the user instead passed an instance of 
        `~halotools.empirical_models.HaloProfileModel`, there is no 
        ambiguity in what is desired: a profile model with parameters 
        that are unbiased with respect to the dark matter halo. 
        So the `interpret_input_model_blueprint` method translates 
        all such instances into `~halotools.empirical_models.GalProfFactory` instances, 
        and returns the appropriately modified blueprint, saving the user 
        a little rigamarole. 
        """

        model_blueprint = copy(self._input_model_blueprint)
        for gal_type in self.gal_types:
            input_prof_model = model_blueprint[gal_type]['profile']
            if isinstance(input_prof_model, halo_prof_components.HaloProfileModel):
                prof_model = gal_prof_factory.GalProfFactory(
                    gal_type, input_prof_model)
                model_blueprint[gal_type]['profile'] = prof_model

        if 'mock_factory' not in model_blueprint.keys():
            model_blueprint['mock_factory'] = mock_factories.HodMockFactory

        return model_blueprint 

    def _set_gal_types(self):
        """ Private method binding the ``gal_types`` list attribute,
        and the ``occupation_bound`` attribute, to the class instance. 

        The ``self.gal_types`` list is sequenced 
        in ascending order of the occupation bound. 
        """

        gal_types = [key for key in self._input_model_blueprint.keys() if key is not 'mock_factory']

        occupation_bounds = []
        for gal_type in gal_types:
            model = self._input_model_blueprint[gal_type]['occupation']
            occupation_bounds.append(model.occupation_bound)

        # Lists have been created. Now sort them and then bind the sorted lists to the instance. 
        sorted_idx = np.argsort(occupation_bounds)
        gal_types = list(np.array(gal_types)[sorted_idx])
        self.gal_types = gal_types

        self.occupation_bound = {}
        for gal_type in self.gal_types:
            self.occupation_bound[gal_type] = (
                self._input_model_blueprint[gal_type]['occupation'].occupation_bound)


    @property 
    def gal_prof_param_list(self):
        """ List of all galaxy profile parameters used by the composite model.

        Notes 
        -----
        Each entry in the list is a string corresponding to a 
        halo profile parameter, but pre-pended by ``gal_``, 
        e.g., ``gal_NFWmodel_conc``.   
        """

        output_list = []
        for gal_type in self.gal_types:
            gal_prof_model = self.model_blueprint[gal_type]['profile']
            output_list.extend(gal_prof_model.gal_prof_func_dict.keys())
        output_list = list(set(output_list))

        return output_list


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

            # Set the method used to return Monte Carlo realizations 
            # of per-halo gal_type abundance
            new_method_name = 'mc_occupation_'+gal_type
            occupation_model = self.model_blueprint[gal_type]['occupation']
            new_method_behavior = partial(occupation_model.mc_occupation, 
                input_param_dict = self.param_dict)
            setattr(self, new_method_name, new_method_behavior)

            # For convenience, also inherit  
            # the first moment of the occupation distribution 
            new_method_name = 'mean_occupation_'+gal_type
            new_method_behavior = partial(occupation_model.mean_occupation, 
                input_param_dict = self.param_dict)
            setattr(self, new_method_name, new_method_behavior)

            ### Now move on to galaxy profiles
            gal_prof_model = self.model_blueprint[gal_type]['profile']

            # Create a new method for each galaxy profile parameter
            # These methods are *not* called by the mock factory
            # See the docstring of composite_gal_prof_param_func 
            # for how the mock factory calls profile models
            for gal_prof_param, func in gal_prof_model.gal_prof_func_dict.iteritems():
                new_method_name = gal_prof_param + '_' + gal_type
                new_method_behavior = func
                setattr(self, new_method_name, new_method_behavior)

            ### Create a method to assign Monte Carlo-realized 
            # positions to each gal_type
            new_method_name = 'pos_'+gal_type
            new_method_behavior = partial(self.mc_pos, gal_type = gal_type)
            setattr(self, new_method_name, new_method_behavior)


        def composite_gal_prof_param_func(gal_prof_param, **kwargs):
            """ Method used to create the function called by the mock factory 
            when assigning profile parameters to galaxy populations. 

            Parameters 
            ----------
            gal_prof_param : string 
                Name of the galaxy profile parameter. 
                Must be equal to one of the galaxy profile parameter names.
                gal_prof_param need not be a profile parameter 
                of the input ``gal_type``. See Notes section. 

            gal_type : string 
                Name of the galaxy population. 

            prim_haloprop : array_like, optional positional argument. 
                See Notes section. 

            sec_haloprop : array_like, optional positional argument. 
                See Notes section. 

            mock_galaxies : object, optional keyword argument 
                See Notes section. 

            Returns 
            -------
            gal_prof_param_func : object
                Function object called by 
                `~halotools.empirical_models.mock_factories.HodMockFactory` 
                to map galaxy profile parameters onto mock galaxies. 

            Notes 
            -----
            The `composite_gal_prof_param_func` is nested within the namespace 
            of `_set_primary_behaviors`, and so it can only be called 
            by `_set_primary_behaviors`. 

            The `_set_primary_behaviors` makes a partial call to this function 
            by passing it only ``gal_prof_param`` as input. In particular, 
            none of the ``mock_galaxies`` inputs are passed. Thus the returned 
            function object takes galaxies as inputs; 
            these inputs can be passed either as arrays or as a collection of mock galaxies. 
            The output of the returned function object is an array of galaxy profile 
            parameters. 

                * Case 1 - ``gal_type`` galaxies have no associated ``gal_prof_param``: the corresonding property of the halo catalog is returned. 
            
                * Case 2 - ``gal_type`` *do* have an associated ``gal_prof_param``: the appropriate `GalProfFactory` is called. 

            """

            gal_type = kwargs['gal_type']
            method_name = gal_prof_param+'_'+gal_type

            if hasattr(self, method_name):
                method_behavior = getattr(self, method_name)
            else:
                halo_prof_param_func_key = (
                    gal_prof_param[len(model_defaults.galprop_prefix):]
                    )
                method_behavior = self.halo_prof_func_dict[halo_prof_param_func_key]

#            return method_behavior(gal_type, **kwargs)
            return method_behavior(**kwargs)

        # Use functools.partial to create a new method of HodModelFactory 
        # by calling composite_gal_prof_param_func, defined above. 
        # See the docstring of composite_gal_prof_param_func 
        # for a description of how this works. 
        for gal_prof_param in self.gal_prof_param_list:
            func = partial(composite_gal_prof_param_func, gal_prof_param)
            setattr(self, gal_prof_param, func)


    def mc_pos(self, mock_obj, **kwargs):
        """ Method used to generate Monte Carlo realizations of galaxy positions. 

        Identical to component model version from which the behavior derives, 
        only this method re-scales the halo-centric distance by the halo radius, 
        and re-centers the re-scaled output of the component model to the halo position.

        Parameters 
        ----------
        mock_obj : object 
            Instance of `~halotools.empirical_models.mock_factories.HodMockFactory`. 

        gal_type : string 
            Name of the galaxy population. 

        Returns 
        -------
        x, y, z : array_like 
            Length-Ngals arrays of coordinate positions, 
            where Ngals is the number of ``gal_type`` gals in the ``mock_obj``. 

        Notes 
        -----
        This method is not directly called by 
        `~halotools.empirical_models.mock_factories.HodMockFactory`. 
        Instead, the `_set_primary_behaviors` method calls functools.partial 
        to create a ``mc_pos_gal_type`` method for each ``gal_type`` in the model. 

        """
        gal_type = kwargs['gal_type']
        gal_prof_model = self.model_blueprint[gal_type]['profile']
        x, y, z = gal_prof_model.mc_pos(mock_obj)

        gal_type_slice = mock_obj._gal_type_indices[gal_type]

        # Re-scale the halo-centric distance by the halo boundary
        if 'halo_boundary' in gal_prof_model.haloprop_key_dict.keys():
            halo_boundary_attr_name = (model_defaults.host_haloprop_prefix + 
                gal_prof_model.haloprop_key_dict['halo_boundary']
                )
        else:
            halo_boundary_attr_name = (
                model_defaults.host_haloprop_prefix + 
                model_defaults.haloprop_key_dict['halo_boundary']
                )

        x *= mock_obj.galaxy_table[halo_boundary_attr_name][gal_type_slice]
        y *= mock_obj.galaxy_table[halo_boundary_attr_name][gal_type_slice]
        z *= mock_obj.galaxy_table[halo_boundary_attr_name][gal_type_slice]

        # Re-center the positions by the host halo location
        halo_xpos_attr_name = model_defaults.host_haloprop_prefix+'x'
        halo_ypos_attr_name = model_defaults.host_haloprop_prefix+'y'
        halo_zpos_attr_name = model_defaults.host_haloprop_prefix+'z'

        x += mock_obj.galaxy_table[halo_xpos_attr_name][gal_type_slice]
        y += mock_obj.galaxy_table[halo_ypos_attr_name][gal_type_slice]
        z += mock_obj.galaxy_table[halo_zpos_attr_name][gal_type_slice]

        return x, y, z


    @property 
    def halo_prof_func_dict(self):
        """ Method to derive the halo profile parameter function 
        dictionary from a collection of component models. 

        Returns 
        -------
        halo_prof_func_dict : dictionary 
            Dictionary storing function objects that specify 
            the mapping between halos and their profile parameters. For details, see the 
            `~halotools.empirical_models.halo_prof_components.HaloProfileModel.halo_prof_func_dict` 
            method of `~halotools.empirical_models.halo_prof_components.HaloProfileModel`. 

        Notes 
        -----
        If there are multiple instances of the same underlying halo profile model, 
        a profile function is effectively chosen at random. 
        This is innocuous, since the multiple instances have already been ensured 
        to provide consistent profile parameter functions. 

        """
        output_halo_prof_func_dict = {}

        for gal_type in self.gal_types:
            halo_prof_model = self.model_blueprint[gal_type]['profile'].halo_prof_model

            for key, func in halo_prof_model.halo_prof_func_dict.iteritems():
                output_halo_prof_func_dict[key] = func

        return output_halo_prof_func_dict


    def build_halo_prof_lookup_tables(self, **kwargs):
        """ Method to create a lookup table 
        used to generate Monte Carlo realizations of 
        radial profiles of galaxies. 

        Parameters 
        ---------- 
        prof_param_table_dict : dict, optional
            Dictionary providing instructions for how to generate a grid of 
            values for each halo profile parameter. 
            Default is an empty dict. For details, see the 
            `~halotools.empirical_models.halo_prof_components.HaloProfileModel.set_prof_param_table_dict`
            method of `~halotools.empirical_models.halo_prof_components.HaloProfileModel`. 

        """
        if 'prof_param_table_dict' in kwargs.keys():
            prof_param_table_dict = kwargs['prof_param_table_dict']
        else:
            prof_param_table_dict = {}

        for gal_type in self.gal_types:
            halo_prof_model = self.model_blueprint[gal_type]['profile'].halo_prof_model
            halo_prof_model.build_inv_cumu_lookup_table(prof_param_table_dict)

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
        for gal_type in self.gal_types:
            gal_type_dict = self.model_blueprint[gal_type]
            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():

                intersection = set(self.param_dict) & set(model_instance.param_dict)
                if intersection != set():
                    repeated_key = list(intersection)[0]
                    raise KeyError("The param_dict key %s appears in more "
                        "than one component model" % repeated_key)
                else:

                    self.param_dict = dict(
                        model_instance.param_dict.items() + 
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

    def _build_composite_lists(self):
        """ A composite model has several lists that are built up from 
        the components: ``_haloprop_list``, ``publications``, and 
        ``new_haloprop_func_dict``. 
        """

        haloprop_list = []
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

                # Reference list
                if hasattr(component_model, 'publications'):
                    pub_list.extend(component_model.publications)

                # Haloprop function dictionaries
                if hasattr(component_model, 'new_haloprop_func_dict'):
                    dict_intersection = set(new_haloprop_func_dict).intersection(
                        set(component_model.new_haloprop_func_dict))
                    if dict_intersection == set():
                        new_haloprop_func_dict = (
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


##########################################










