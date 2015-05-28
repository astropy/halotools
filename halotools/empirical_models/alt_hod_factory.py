# -*- coding: utf-8 -*-
"""
Module containing the primary class used to build 
composite HOD-style models from a set of components. 
"""

__all__ = ['AltHodModelFactory']
__author__ = ['Andrew Hearin']

from functools import partial
import numpy as np
from copy import copy

from . import occupation_helpers as occuhelp
from . import model_defaults
from . import mock_factory
from . import preloaded_hod_blueprints
from . import gal_prof_factory
from . import halo_prof_components

from ..sim_manager.read_nbody import ProcessedSnapshot
from ..sim_manager.generate_random_sim import FakeSim


class AltHodModelFactory(object):
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

    Examples 
    --------
    The simplest way to build an HOD-style model is to use one of the pre-loaded blueprints. 
    Let's use `Kravtsov04` as a simple example:

    >>> blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint()
    >>> model = HodModelFactory(blueprint)

    Now let's populate a simulation using our newly created model object. 
    The `~halotools.sim_manager` sub-package contains methods that let you choose from a 
    range of publicly available N-body simulations into which you can sprinkle mock galaxies 
    with your model. For these demonstration purposes, we'll use a fake simulation: 

    >>> fake_snapshot = FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot)
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

        # Bind the model-building instructions to the composite model
        self._input_model_blueprint = input_model_blueprint

        # Create attributes for galaxy types and their occupation bounds
        self._set_gal_types()
        self.model_blueprint = self.interpret_input_model_blueprint()

        # Build the composite model dictionary, 
        # whose keys are parameters of our model
        self._set_init_param_dict()

        # Create a set of bound methods with specific names 
        # that will be called by the mock factory 
        self._set_primary_behaviors()

        self.publications = self._build_publication_list()


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
            model_blueprint['mock_factory'] = mock_factory.HodMockFactory

        return model_blueprint 


    def populate_mock(self, **kwargs):
        """ Method used to populate a simulation using the model. 

        After calling this method, ``self`` will have a new ``mock`` attribute, 
        which is an instance of `~halotools.empirical_models.mock_factory.HodMockFactory`. 

        Parameters 
        ----------
        snapshot : object, optional keyword argument
            Class instance of `~halotools.sim_manager.ProcessedSnapshot`. 
            This object contains the halo catalog and its metadata.  

        kwargs : additional optional keyword arguments 
            Any keyword of either 
            `~halotools.sim_manager.read_nbody.ProcessedSnapshot` or 
            `~halotools.empirical_models.mock_factory.HodMockFactory` is supported. 
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


    @property 
    def haloprop_key_dict(self):
        """ Dictionary defining the halo properties 
        that regulate galaxy occupation statistics. 

        Dict keys always include ``prim_haloprop_key`` and ``halo_boundary``, 
        whose default settings are defined in `~halotools.empirical_models.model_defaults`. 
        Models with assembly bias will include a ``sec_haloprop_key`` key. 
        Dict values are strings used to access the appropriate column of a halo catalog, 
        e.g., ``mvir``. 
        """

        return return_haloprop_dict(self.model_blueprint)

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
            # See docstring of _get_gal_prof_param for how this works
            # These methods are *not* called by the mock factory
            # See the docstring of composite_gal_prof_param_func 
            # for how the mock factory calls profile models
            for gal_prof_param in gal_prof_model.gal_prof_func_dict.keys():
                new_method_name = gal_prof_param + '_' + gal_type
                new_method_behavior = partial(self._get_gal_prof_param, 
                    gal_prof_param, gal_type)
                setattr(self, new_method_name, new_method_behavior)

            ### Create a method to assign Monte Carlo-realized 
            # positions to each gal_type
            new_method_name = 'pos_'+gal_type
            new_method_behavior = partial(self.mc_pos, gal_type = gal_type)
            setattr(self, new_method_name, new_method_behavior)


        def composite_gal_prof_param_func(gal_prof_param, gal_type, *args, **kwargs):
            """ Method used to create the function called by the mock factory 
            when assigning profile parameters to galaxy populations. 

            Parameters 
            ----------
            gal_prof_param : string 
                Name of the galaxy profile parameter. 
                Must be equal to one of the galaxy profile parameter names.
                Unlike `_get_gal_prof_param`, ``gal_prof_param`` need not 
                be a profile parameter of the input ``gal_type``. See Notes section. 

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
                `~halotools.empirical_models.mock_factory.HodMockFactory` 
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
            
                * Case 2 - ``gal_type`` *do* have an associated ``gal_prof_param``: the appropriate `GalProfFactory` is called. This behavior has already been set by the `_set_primary_behaviors` call to `_get_gal_prof_param`. 

            """

            method_name = gal_prof_param+'_'+gal_type

            if hasattr(self, method_name):
                method_behavior = getattr(self, method_name)
            else:
                halo_prof_param_func_key = (
                    gal_prof_param[len(model_defaults.galprop_prefix):]
                    )
                method_behavior = self.halo_prof_func_dict[halo_prof_param_func_key]

            return method_behavior(*self.retrieve_relevant_haloprops(
                gal_type, *args, **kwargs))

        # Use functools.partial to create a new method of HodModelFactory 
        # by calling composite_gal_prof_param_func, defined above. 
        # See the docstring of composite_gal_prof_param_func 
        # for a description of how this works. 
        for gal_prof_param in self.gal_prof_param_list:
            func = partial(composite_gal_prof_param_func, gal_prof_param)
            setattr(self, gal_prof_param, func)


    def _get_gal_prof_param(self, gal_prof_param, gal_type, *args, **kwargs):
        """ Private method used by `_set_primary_behaviors` to assign (possibly biased) 
        profile parameters to mock galaxies. 

        Parameters 
        ----------
        gal_prof_param : string 
            Name of the galaxy profile parameter. 
            Must be equal to one of the galaxy profile parameter names.
            For example, if the input ``gal_type`` pertains to 
            a satellite-like population tracing a (possibly biased) NFW profile, 
            then ``gal_prof_param`` would be ``gal_NFWmodel_conc``. 

        gal_type : string 
            Name of the galaxy population. 

        prim_haloprop : array_like, optional positional argument 
            See Notes section. 

        sec_haloprop : array_like, optional positional argument 
            See Notes section. 

        mock_galaxies : object, optional keyword argument 
            See Notes section. 

        input_param_dict : optional keyword argument 

        Returns
        -------
        gal_prof_param_func : function object 
            Function used to map values of ``gal_prof_param`` onto 
            ``gal_type`` galaxies. 

        Notes 
        -----
        Must pass either ``prim_haloprop``, or ``mock_galaxies``, but not both, 
        and not neither. If model has assembly-biased spatial positions, 
        and if not passing ``mock_galaxies``, but pass both 
        ``prim_haloprop`` and ``sec_haloprop``. 

        `_get_gal_prof_param` is used exclusively by `_set_primary_behaviors`, 
        which calls `_get_gal_prof_param` by passing it *only* the following 
        two positional arguments: ``gal_prof_param`` and ``gal_type``. 
        Thus the input of the function object returned by `_get_gal_prof_param` 
        only takes ``mock_galaxies`` as input; 
        these inputs can be passed either as arrays or as a collection of mock galaxies. 
        The output of the function object returned by `_get_gal_prof_param` 
        is the array of profile parameter values ``gal_prof_param`` 
        that pertain to ``gal_type`` galaxies. 

        """
        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
        else:
            input_param_dict = {}

        gal_prof_model = self.model_blueprint[gal_type]['profile']
        gal_prof_param_func = partial(
            gal_prof_model.gal_prof_func_dict[gal_prof_param], 
            input_param_dict = input_param_dict)

        return gal_prof_param_func(
            *self.retrieve_relevant_haloprops(gal_type, *args, **kwargs))


    def mc_pos(self, mock_galaxies, gal_type):
        """ Method used to generate Monte Carlo realizations of galaxy positions. 

        Identical to component model version from which the behavior derives, 
        only this method re-scales the halo-centric distance by the halo radius, 
        and re-centers the re-scaled output of the component model to the halo position.

        Parameters 
        ----------
        mock_galaxies : object 
            Collection of mock galaxies created by 
            `~halotools.empirical_models.mock_factory.HodMockFactory`. 

        gal_type : string 
            Name of the galaxy population. 

        Returns 
        -------
        x, y, z : array_like 
            Length-Ngals arrays of coordinate positions, 
            where Ngals is the number of ``gal_type`` gals in the ``mock_galaxies``. 

        Notes 
        -----
        This method is not directly called by 
        `~halotools.empirical_models.mock_factory.HodMockFactory`. 
        Instead, the `_set_primary_behaviors` method calls functools.partial 
        to create a ``mc_pos_gal_type`` method for each ``gal_type`` in the model. 

        """
        gal_prof_model = self.model_blueprint[gal_type]['profile']
        mc_pos_function = getattr(gal_prof_model, 'mc_pos')

        x, y, z = mc_pos_function(mock_galaxies)

        gal_type_slice = mock_galaxies._gal_type_indices[gal_type]

        # Re-scale the halo-centric distance by the halo boundary
        halo_boundary_attr_name = (
            model_defaults.host_haloprop_prefix + 
            model_defaults.haloprop_key_dict['halo_boundary']
            )

        x *= getattr(mock_galaxies, halo_boundary_attr_name)[gal_type_slice]
        y *= getattr(mock_galaxies, halo_boundary_attr_name)[gal_type_slice]
        z *= getattr(mock_galaxies, halo_boundary_attr_name)[gal_type_slice]

        # Re-center the positions by the host halo location
        halo_xpos_attr_name = model_defaults.host_haloprop_prefix+'x'
        halo_ypos_attr_name = model_defaults.host_haloprop_prefix+'y'
        halo_zpos_attr_name = model_defaults.host_haloprop_prefix+'z'

        x += getattr(mock_galaxies, halo_xpos_attr_name)[gal_type_slice]
        y += getattr(mock_galaxies, halo_ypos_attr_name)[gal_type_slice]
        z += getattr(mock_galaxies, halo_zpos_attr_name)[gal_type_slice]

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

                occuhelp.test_repeated_keys(
                    self.param_dict, model_instance.param_dict)

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
        self._set_init_param_dict()
        self._set_primary_behaviors()

    def _build_publication_list(self):
        """ Method to build a list of publications 
        associated with each component model. 

        Parameters 
        ----------
        model_blueprint : dict 
            Dictionary passed to the HOD factory __init__ constructor 
            that is used to provide instructions for how to build a 
            composite model from a set of components. 

        Returns 
        -------
        pub_list : array_like 
        """
        pub_list = []

        # Loop over all galaxy types in the composite model
        for gal_type in self.gal_types:
            gal_type_dict = self.model_blueprint[gal_type]

            # For each galaxy type, loop over its features
            for model_instance in gal_type_dict.values():
                if hasattr(model_instance, 'publications'):
                    pub_list.extend(model_instance.publications)

        return list(set(pub_list))

##########################################





