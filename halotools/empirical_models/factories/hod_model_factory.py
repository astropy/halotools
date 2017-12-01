"""
Module storing the `~halotools.empirical_models.HodModelFactory`,
the primary factory responsible for building
HOD-style models of the galaxy-halo connection.
"""

import numpy as np
from copy import copy
from warnings import warn
import collections

from .model_factory_template import ModelFactory
from .hod_mock_factory import HodMockFactory

from .. import model_helpers

from ..occupation_models import OccupationComponent
from ...sim_manager import sim_defaults
from ...custom_exceptions import HalotoolsError

__all__ = ['HodModelFactory']
__author__ = ['Andrew Hearin']


class HodModelFactory(ModelFactory):
    """ Class used to build HOD-style models of the galaxy-halo connection.

    See :ref:`hod_modeling_tutorial0` for an in-depth description
    of how to build HOD models, demonstrated by a
    sequence of increasingly complex examples.
    If you do not wish to build your own model but want to use one
    provided by Halotools,
    instead see `~halotools.empirical_models.PrebuiltHodModelFactory`.

    All HOD-style composite models can directly populate catalogs of dark matter halos.
    For an in-depth description of how Halotools implements this mock-generation, see
    :ref:`hod_mock_factory_source_notes`.

    The arguments passed to the `HodModelFactory` constructor determine
    the features of the model that are returned by the factory. This works in one of two ways,
    both of which have explicit examples provided below.

    1. Building a new model from scratch.

    You can build a model from scratch by passing in a sequence of
    ``model_features``, each of which are instances of component models.
    The factory then composes these independently-defined
    components into a composite model.

    2. Building a new model from an existing model.

    It is also possible to add/swap new features to a previously built composite model instance,
    allowing you to create new models from existing ones. To do this, you pass in
    a ``baseline_model_instance`` and any set of ``model_features``.
    Any ``model_feature`` keyword that matches a feature name of the ``baseline_model_instance``
    will replace that feature in the ``baseline_model_instance``;
    all other ``model_features`` that you pass in will augment
    the ``baseline_model_instance`` with new behavior.

    Regardless what set of features you use to build your model,
    the returned object can be used to directly populate a halo catalog
    with mock galaxies using the
    `~halotools.empirical_models.HodModelFactory.populate_mock` method,
    as shown in the example below.

    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------

        *model_features : sequence of keyword arguments, optional
            Each keyword you use will be interpreted as the name
            of a feature in the composite model,
            e.g. 'stellar_mass' or 'star_formation_rate';
            the value bound to each keyword must be an instance of a
            component model governing the behavior of that feature.
            See the examples section below.

        baseline_model_instance : `SubhaloModelFactory` instance, optional
            If passed to the constructor, the ``model_dictionary`` bound to the
            ``baseline_model_instance`` will be treated as the baseline dictionary.
            Any additional keyword arguments passed to the constructor that appear
            in the baseline dictionary will be treated as model features that replace
            the corresponding component model in the baseline dictionary. Any
            model features passed to the constructor that do not
            appear in the baseline dictionary will be treated as new features that
            augment the baseline model with new behavior. See the examples section below.

        model_feature_calling_sequence : list, optional
            Determines the order in which your component features
            will be called during mock population.

            Some component models may have explicit dependence upon
            the value of some other galaxy property being modeled.
            In such a case, you must pass a ``model_feature_calling_sequence`` list,
            ordered in the desired calling sequence.

            A classic example is if the stellar-to-halo-mass relation
            has explicit dependence on the star formation rate of the galaxy
            (active or quiescent). For this example, the
            ``model_feature_calling_sequence`` would be
            model_feature_calling_sequence = ['sfr_designation', 'stellar_mass', ...].

            Default behavior is to assume that no model feature
            has explicit dependence upon any other, in which case the component
            models appearing in the ``model_features`` keyword arguments
            will be called in random order, giving primacy to the potential presence
            of `stellar_mass` and/or `luminosity` features.

        gal_type_list : list, optional
            List of strings providing the names of the galaxy types in the
            composite model. This is only necessary to provide if you have
            a gal_type in your model that is neither ``centrals`` nor ``satellites``.

            For example, if you have entirely separate models for ``red_satellites`` and
            ``blue_satellites``, then your ``gal_type_list`` might be,
            gal_type_list = ['centrals', 'red_satellites', 'blue_satellites'].
            Another possible example would be
            gal_type_list = ['centrals', 'satellites', 'orphans'].

        redshift: float, optional
            Redshift of the model galaxies. Must be compatible with the
            redshift of all component models, and with the redshift
            of the snapshot of the simulation used to populate mocks.
            Default is None.

        halo_selection_func : function object, optional
            Function object used to place a cut on the input ``table``.
            If the ``halo_selection_func`` keyword argument is passed,
            the input to the function must be a single positional argument storing a
            length-N structured numpy array or Astropy table;
            the function output must be a length-N boolean array that will be used as a mask.
            Halos that are masked will be entirely neglected during mock population.

        Examples
        ---------
        As described above, there are two different ways to build models using the
        `HodModelFactory`. Here we give demonstrations of each in turn.

        In the first example we'll show how to build a model from scratch using
        the ``model_features`` option. For illustration purposes, we'll pick a
        particularly simple HOD-style model based on Zheng et al. (2007). As
        described in `~halotools.empirical_models.zheng07_model_dictionary`, in this model
        there are two galaxy populations, 'centrals' and 'satellites';
        centrals sit at the center of dark matter halos, and satellites follow an NFW profile.

        We'll start with the features for the population of centrals:

        >>> from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
        >>> cens_occ_model =  Zheng07Cens()
        >>> cens_prof_model = TrivialPhaseSpace()

        Now for the satellites:

        >>> from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
        >>> sats_occ_model =  Zheng07Sats()
        >>> sats_prof_model = NFWPhaseSpace()

        At this point we have our component model instances.
        The following call to the factory uses the ``model_features`` option
        described above:

        >>> model_instance = HodModelFactory(centrals_occupation = cens_occ_model, centrals_profile = cens_prof_model, satellites_occupation = sats_occ_model, satellites_profile = sats_prof_model)

        The feature names we have chosen are 'centrals_occupation' and 'centrals_profile',
        'satellites_occupation' and 'satellites_profile'. The first substring of each feature name
        informs the factory of the name of the galaxy population, the second substring identifies
        the type of feature; to each feature we have attached a component model instance.

        Whatever features your composite model has,
        you can use the `~HodModelFactory.populate_mock` method
        to create Monte Carlo realization of the model by populating any dark matter halo catalog
        in your cache directory:

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0.5) # doctest: +SKIP
        >>> model_instance.populate_mock(halocat) # doctest: +SKIP

        Your ``model_instance`` now has a ``mock`` attribute storing a synthetic galaxy
        population. See the `~HodModelFactory.populate_mock` docstring for details.

        There also convenience functions for estimating the clustering signal predicted by the model.
        For example, the following method repeatedly populates the Bolshoi simulation with
        galaxies, computes the 3-d galaxy clustering signal of each mock, computes the median
        clustering signal in each bin, and returns the result:

        >>> r, xi = model_instance.compute_average_galaxy_clustering(num_iterations = 5, simname = 'bolshoi', redshift = 0.5) # doctest: +SKIP

        In this next example we'll show how to build a new model from an existing one
        using the ``baseline_model_instance`` option. We will start from
        the composite model built in Example 1 above. Here we'll build a
        new model which is identical the ``model_instance`` above,
        only we instead use
        the `AssembiasZheng07Cens` class to introduce assembly bias into the
        occupation statistics of central galaxies.

        >>> from halotools.empirical_models import AssembiasZheng07Cens
        >>> new_cen_occ_model = AssembiasZheng07Cens()
        >>> new_model_instance = HodModelFactory(baseline_model_instance = model_instance, centrals_occupation = new_cen_occ_model)

        The ``new_model_instance`` and the original ``model_instance`` are identical in every respect
        except for the assembly bias of central galaxy occupation.

        See also
        ---------
        :ref:`hod_model_factory_source_notes`

        :ref:`hod_mock_factory_source_notes`

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
            #  Making a copy is not strictly necessary, but we do it here to emphasize
            #  at the syntax-level that the model_dictionary and _input_model_dictionary
            #  are fully independent, not pointers to the same locations in memory
            self.model_dictionary[key] = copy(self._input_model_dictionary[key])

        self._test_censat_occupation_consistency(self.model_dictionary)

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
        """ Method used to parse the arguments passed to
        the constructor into a model dictionary and supplementary arguments.

        `parse_constructor_kwargs` examines the keyword arguments passed to `__init__`,
        and identifies the possible presence of ``galaxy_selection_func``,
        ``halo_selection_func``, ``model_feature_calling_sequence`` and ``gal_type_list``;
        all other keyword arguments will be treated as component models,
        and it is enforced that the values bound to all such arguments
        at the very least have a ``_methods_to_inherit`` attribute.

        Parameters
        -----------
        **kwargs : optional keyword arguments
            keywords will be interpreted as the ``feature name``;
            values must be instances of Halotools component models

        Returns
        --------
        input_model_dictionary : dict
            Model dictionary defining the composite model.

        supplementary_kwargs : dict
            Dictionary of any possible remaining keyword arguments passed to the `__init__` constructor
            that are not part of the composite model dictionary, e.g., ``model_feature_calling_sequence``.
        """
        if len(kwargs) == 0:
            msg = ("You did not pass any model features to the factory")
            raise HalotoolsError(msg)

        try:
            self._factory_constructor_redshift = kwargs.pop('redshift')
        except KeyError:
            pass

        if 'baseline_model_instance' in kwargs:
            baseline_model_dictionary = kwargs['baseline_model_instance'].model_dictionary
            input_model_dictionary = copy(kwargs)
            del input_model_dictionary['baseline_model_instance']

            # First parse the supplementary keyword arguments,
            # such as 'model_feature_calling_sequence',
            # from the keywords that are bound to component model instances,
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

            new_model_dictionary = copy(baseline_model_dictionary)
            for key, value in input_model_dictionary.items():
                new_model_dictionary[key] = value
            return new_model_dictionary, supplementary_kwargs

        else:
            input_model_dictionary = copy(kwargs)

            # First parse the supplementary keyword arguments,
            # such as 'model_feature_calling_sequence',
            # from the keywords that are bound to component model instances,
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
        """ Method uses the ``model_feature_calling_sequence`` passed to __init__, if available.
        If no such argument was passed, the default sequence
        will be to first call ``occupation`` features, then call all other features in a random order,
        always calling features associated with a ``centrals`` population first (if presesent).

        Parameters
        -----------
        supplementary_kwargs : dict
            Dictionary storing all keyword arguments passed to the `__init__` constructor that were
            not part of the input model dictionary.

        Returns
        -------
        model_feature_calling_sequence : list
            List of strings specifying the order in which the component models will be called upon
            during mock population to execute their methods.

        See also
        ---------
        :ref:`model_feature_calling_sequence_mechanism`
        """

        ########################
        # Require that all elements of the input model_feature_calling_sequence
        # were also keyword arguments to the __init__ constructor
        try:
            model_feature_calling_sequence = list(supplementary_kwargs['model_feature_calling_sequence'])
            for model_feature in model_feature_calling_sequence:
                try:
                    assert model_feature in list(self._input_model_dictionary.keys())
                except AssertionError:
                    msg = ("\nYour input ``model_feature_calling_sequence`` has a ``%s`` element\n"
                    "that does not appear in the keyword arguments you passed to the HodModelFactory.\n"
                    "For every element of the input ``model_feature_calling_sequence``, "
                    "there must be a corresponding \n"
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
        # Now conversely require that all remaining __init__ constructor keyword arguments
        # appear in the model_feature_calling_sequence
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
                    "to the constructor of the HodModelFactory, "
                    "\nfrom which it was inferred that your intended"
                    "``gal_type`` = %s, which is inconsistent.\n"
                    "If this inferred ``gal_type`` seems incorrect,\n"
                    "please raise an Issue on https://github.com/astropy/halotools.\n"
                    "Otherwise, either change the ``%s`` keyword argument "
                    "to conform to the Halotools convention \n"
                    "to use keyword arguments that are composed of a "
                    "``gal_type`` and ``feature_name`` substring,\n"
                    "separated by a '_', in that order.\n")
                raise HalotoolsError(msg %
                    (component_model_class_name, component_model_gal_type,
                        model_feature_calling_sequence_element,
                        gal_type, model_feature_calling_sequence_element))

            try:
                assert feature_name == component_model_feature_name
            except AssertionError:
                msg = ("\nThe ``%s`` component model instance has ``feature_name`` = %s.\n"
                    "However, you used a keyword argument = ``%s`` when passing this component model \n"
                    "to the constructor of the HodModelFactory, \nfrom which it was inferred that your intended"
                    "``feature_name`` = %s, which is inconsistent.\n"
                    "If this inferred ``feature_name`` seems incorrect,\n"
                    "please raise an Issue on https://github.com/astropy/halotools.\n"
                    "Otherwise, either change the ``%s`` keyword argument "
                    "to conform to the Halotools convention \n"
                    "to use keyword arguments that are composed of a "
                    "``gal_type`` and ``feature_name`` substring,\n"
                    "separated by a '_', in that order.\n")
                raise HalotoolsError(msg %
                    (component_model_class_name, component_model_feature_name,
                        model_feature_calling_sequence_element,
                        feature_name, model_feature_calling_sequence_element))

    def _infer_gal_type_and_feature_name(self, model_dictionary_key, gal_type_list,
            known_gal_type=None, known_feature_name=None):

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
                        processed_key, gal_type_guess_list, known_gal_type=known_gal_type)
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
        for component_model in list(self.model_dictionary.values()):
            _gal_type_list.append(component_model.gal_type)
        self.gal_types = sorted(list(set(_gal_type_list)))

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

                docstring = getattr(component_model, methodname).__doc__
                getattr(self, new_method_name).__doc__ = docstring

                if hasattr(component_model, '_additional_kwargs_dict'):
                    additional_kwargs_dict = component_model._additional_kwargs_dict
                    self._test_additional_kwargs_dict(additional_kwargs_dict)
                    try:
                        additional_kwargs = additional_kwargs_dict[methodname]
                        setattr(getattr(self, new_method_name),
                            'additional_kwargs', additional_kwargs)
                    except KeyError:
                        pass

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

        The behavior of the methods bound to the composite model are decorated versions
        of the methods defined in the component models. The decoration is done with
        `update_param_dict_decorator`. For each function that gets bound to the
        composite model, what this decorator does is search the param_dict of the
        component_model associated with the function, and update all matching keys
        in that param_dict with the param_dict of the composite.
        This way, all the user needs to do is make changes to the composite model
        param_dict. Then, when calling any method of the composite model,
        the changed values of the param_dict automatically propagate down
        to the component model before calling upon its behavior.
        This allows the composite_model to control behavior
        of functions that it does not define.

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

        for component_model in list(self.model_dictionary.values()):
            if hasattr(component_model, 'build_lookup_tables'):
                component_model.build_lookup_tables()

    def build_init_param_dict(self):
        """ Create the ``param_dict`` attribute of the instance. The ``param_dict`` is a dictionary storing
        the full collection of parameters controlling the behavior of the composite model.

        The ``param_dict`` dictionary is determined by examining the
        ``param_dict`` attribute of every component model, and building up a composite
        dictionary from them. It is permissible for the same parameter name to appear more than once
        amongst a set of component models, but a warning will be issued in such cases.

        Notes
        -----
        In MCMC applications, the items of ``param_dict`` defines the possible
        parameter set explored by the likelihood engine.
        Changing the values of the parameters in ``param_dict``
        will propagate to the behavior of the component models
        when the relevant methods are called.

        See also
        ---------
        set_warning_suppressions

        :ref:`param_dict_mechanism`

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

        for component_model in list(self.model_dictionary.values()):

            if not hasattr(component_model, 'param_dict'):
                component_model.param_dict = {}
            intersection = set(self.param_dict.keys()) & set(component_model.param_dict.keys())
            if intersection != set():
                for key in intersection:
                    if suppress_warning is False:
                        warn(msg % key)

            for key, value in component_model.param_dict.items():
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

        zlist = list(model.redshift for model in list(self.model_dictionary.values())
            if hasattr(model, 'redshift'))

        if len(set(zlist)) == 0:
            try:
                self.redshift = self._factory_constructor_redshift
            except AttributeError:
                self.redshift = sim_defaults.default_redshift
        elif len(set(zlist)) == 1:
            self.redshift = float(zlist[0])
        else:
            msg = ("Inconsistency between the redshifts of the component models:\n\n")
            for model in list(self.model_dictionary.values()):
                gal_type = model.gal_type
                clname = model.__class__.__name__
                if hasattr(model, 'redshift'):
                    zs = str(model.redshift)
                    msg += ("For gal_type = ``" + gal_type + "``, the " +
                        clname+" instance has redshift = " + zs + "\n")
            raise HalotoolsError(msg)

        if hasattr(self, '_factory_constructor_redshift'):
            msg = ("You passed an input argument of ``redshift`` = {0} to the HodModelFactory\n"
                "that is inconsistent with the redshift z = {1} defined by "
                "the component models".format(
                    self._factory_constructor_redshift, self.redshift))
            assert self.redshift == self._factory_constructor_redshift, msg

    def build_prim_sec_haloprop_list(self):
        """ Method builds the ``_haloprop_list`` of strings.

        This list stores the names of all halo catalog columns
        that appear as either ``prim_haloprop_key`` or ``sec_haloprop_key`` of any component model.
        For all strings appearing in ``_haloprop_list``, the mock ``galaxy_table`` will have
        a corresponding column storing the halo property inherited by the mock galaxy.
        """
        haloprop_list = []
        for component_model in list(self.model_dictionary.values()):

            if hasattr(component_model, 'prim_haloprop_key'):
                haloprop_list.append(component_model.prim_haloprop_key)
            if hasattr(component_model, 'sec_haloprop_key'):
                haloprop_list.append(component_model.sec_haloprop_key)
            if hasattr(component_model, 'halo_boundary_key'):
                haloprop_list.append(component_model.halo_boundary_key)
            if hasattr(component_model, 'list_of_haloprops_needed'):
                haloprop_list.extend(component_model.list_of_haloprops_needed)

        self._haloprop_list = list(set(haloprop_list))

    def build_prof_param_keys(self):
        """
        """
        halo_prof_param_keys = []
        gal_prof_param_keys = []

        for component_model in list(self.model_dictionary.values()):
            if hasattr(component_model, 'halo_prof_param_keys'):
                halo_prof_param_keys.extend(component_model.halo_prof_param_keys)
            if hasattr(component_model, 'gal_prof_param_keys'):
                gal_prof_param_keys.extend(component_model.gal_prof_param_keys)

        self.halo_prof_param_keys = list(set(halo_prof_param_keys))
        self.gal_prof_param_keys = list(set(gal_prof_param_keys))

    def build_publication_list(self):
        """
        """
        pub_list = []
        for component_model in list(self.model_dictionary.values()):

            if hasattr(component_model, 'publications'):
                pub_list.extend(component_model.publications)

        self.publications = list(set(pub_list))

    def build_dtype_list(self):
        """ Create the `_galprop_dtypes_to_allocate` attribute that determines
        the name and data type of every galaxy property that will appear in the mock ``galaxy_table``.

        This attribute is determined by examining the
        `_galprop_dtypes_to_allocate` attribute of every component model, and building a composite
        set of all these dtypes, enforcing self-consistency in cases where the same galaxy property
        appears more than once.

        See also
        ---------
        :ref:`galprop_dtypes_to_allocate_mechanism`
        """
        dtype_list = []
        for component_model in list(self.model_dictionary.values()):

            # Column dtypes to add to mock galaxy_table
            if hasattr(component_model, '_galprop_dtypes_to_allocate'):
                dtype_list.append(component_model._galprop_dtypes_to_allocate)

        self._galprop_dtypes_to_allocate = model_helpers.create_composite_dtype(dtype_list)

    def build_new_haloprop_func_dict(self):
        """ Method used to build a dictionary of functions, ``new_haloprop_func_dict``,
        that create new halo catalog columns
        during a pre-processing phase of mock population.

        See also
        ---------
        :ref:`new_haloprop_func_dict_mechanism`
        """
        new_haloprop_func_dict = {}

        for component_model in list(self.model_dictionary.values()):
            feature_name, gal_type = component_model.feature_name, component_model.gal_type

            # Haloprop function dictionaries
            if hasattr(component_model, 'new_haloprop_func_dict'):
                dict_intersection = set(new_haloprop_func_dict).intersection(
                    set(component_model.new_haloprop_func_dict))
                if dict_intersection == set():
                    new_haloprop_func_dict = dict(
                        list(new_haloprop_func_dict.items()) +
                        list(component_model.new_haloprop_func_dict.items())
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
        """ Method used to determine whether a warning should be issued if the
        `build_init_param_dict` method detects the presence of multiple appearances
        of the same parameter name.

        If *any* of the component model instances have a
        ``_suppress_repeated_param_warning`` attribute that is set to the boolean True value,
        then no warning will be issued even if there are multiple appearances of the same
        parameter name. This allows the user to not be bothered with warning messages for cases
        where it is understood that there will be no conflicting behavior.

        See also
        ---------
        build_init_param_dict
        """
        self._suppress_repeated_param_warning = False

        for component_model in list(self.model_dictionary.values()):

            if hasattr(component_model, '_suppress_repeated_param_warning'):
                self._suppress_repeated_param_warning += component_model._suppress_repeated_param_warning

    def set_inherited_methods(self):
        """ Each component model *should* have a ``_mock_generation_calling_sequence`` attribute
        that provides the sequence of method names to call during mock population. Additionally,
        each component *should* have a ``_methods_to_inherit`` attribute that determines
        which methods will be inherited by the composite model.
        The ``_mock_generation_calling_sequence`` list *should* be a subset of ``_methods_to_inherit``.
        If any of the above conditions fail, no exception will be raised during the construction
        of the composite model. Instead, an empty list will be forcibly attached to each
        component model for which these lists may have been missing.
        Also, for each component model, if there are any elements of ``_mock_generation_calling_sequence``
        that were missing from ``_methods_to_inherit``, all such elements will be forcibly added to
        that component model's ``_methods_to_inherit``.

        Finally, each component model *should* have an ``_attrs_to_inherit`` attribute that determines
        which attributes will be inherited by the composite model. If any component models did not
        implement the ``_attrs_to_inherit``, an empty list is forcibly added to the component model.

        After calling the set_inherited_methods method, it will be therefore be entirely safe to
        run a for loop over each component model's ``_methods_to_inherit`` and ``_attrs_to_inherit``,
        even if these lists were forgotten or irrelevant to that particular component.
        """

        for component_model in list(self.model_dictionary.values()):

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

            missing_methods = (set(mock_making_methods) -
                set(inherited_methods).intersection(set(mock_making_methods)))
            for methodname in missing_methods:
                component_model._methods_to_inherit.append(methodname)

            if not hasattr(component_model, '_attrs_to_inherit'):
                component_model._attrs_to_inherit = []

    def set_calling_sequence(self):
        """ Method used to determine the sequence of function calls that will be made during
        mock population. The methods of each component model will be called one after the other;
        the order in which the component models are called upon is determined by
        ``_model_feature_calling_sequence``.
        When each component model is called, the sequence of methods that are called for that
        component is determined by the ``_mock_generation_calling_sequence`` attribute
        bound to the component model instance.
        See :ref:`model_feature_calling_sequence_mechanism` for further details.
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
                msg = ("Inconsistency in the threshold of the "
                    "component occupation models:\n" + threshold_msg + "\n")
                raise HalotoolsError(msg)

        missing_method_msg1 = ("\nAll component models have a "
            "``_mock_generation_calling_sequence`` attribute,\n"
            "which is a list of method names that are called by the "
            "``populate_mock`` method of the mock factory.\n"
            "All component models also have a ``_methods_to_inherit`` attribute, \n"
            "which determines which methods of the component model are inherited by the composite model.\n"
            "The former must be a subset of the latter. However, for ``gal_type`` = %s,\n"
            "the following method was not inherited:\n%s")
        for component_model in list(self.model_dictionary.values()):

            mock_generation_methods = set(component_model._mock_generation_calling_sequence)
            inherited_methods = set(component_model._methods_to_inherit)
            overlap = mock_generation_methods.intersection(inherited_methods)
            missing_methods = mock_generation_methods - overlap
            if missing_methods != set():
                some_missing_method = list(missing_methods)[0]
                raise HalotoolsError(missing_method_msg1 % (component_model.gal_type, some_missing_method))

        missing_method_msg2 = ("\nAll component models have a "
            "``_mock_generation_calling_sequence`` attribute,\n"
            "which is a list of method names that are called by the "
            "``populate_mock`` method of the mock factory.\n"
            "The HodModelFactory builds a composite ``_mock_generation_calling_sequence`` "
            "from each of these lists.\n"
            "However, the following method does not appear to have been created during this process:\n%s\n"
            "This is likely a bug in Halotools - "
            "please raise an Issue on https://github.com/astropy/halotools\n")
        for method in self._mock_generation_calling_sequence:
            if not hasattr(self, method):
                raise HalotoolsError(missing_method_msg2)

    def _test_censat_occupation_consistency(self, model_dictionary):
        """ This private method searches each OccupationComponent instance for a
        ``central_occupation_model`` attribute. If detected, a check is made on the
        self-consistency between the class of the object bound to that attribute,
        and the class of the occupation model actually bound to the centrals population.
        """
        occu_model_list = list(obj for obj in model_dictionary.values()
            if isinstance(obj, OccupationComponent))

        actual_cenocc_model_exists = False
        for i, occu_model in enumerate(occu_model_list):
            try:
                gal_type = occu_model.gal_type
                if gal_type == 'centrals':
                    actual_cenocc_model = occu_model
                    actual_cenocc_model_exists = True
            except AttributeError:
                pass

        if not actual_cenocc_model_exists:
            # There is no central occupation model to be inconsistent with
            return
        else:
            for component_model in occu_model_list:
                try:
                    subordinate_cenocc_model = getattr(component_model, 'central_occupation_model')
                    assert isinstance(subordinate_cenocc_model, actual_cenocc_model.__class__)
                    try:
                        assert set(subordinate_cenocc_model.param_dict) == set(actual_cenocc_model.param_dict)
                    except AttributeError:
                        raise HalotoolsError("The ``centrals`` occupation model "
                            "must have a ``param_dict`` attribute\n")
                except AttributeError:
                    pass
                except AssertionError:
                    msg = ("The occupation component of gal_type = ``{0}`` galaxies \n"
                        "has a ``central_occupation_model`` attribute with an inconsistent \n"
                        "implementation with the {1} class controlling the "
                        "occupation statistics of the ``centrals`` population.\n"
                        "If you use the ``cenocc_model`` feature, you must build a \n"
                        "composite model with a self-consistent population of centrals.\n".format(
                            component_model.gal_type, component_model.__class__.__name__))
                    raise HalotoolsError(msg)

    def _test_additional_kwargs_dict(self, _additional_kwargs_dict):
        """
        """
        assert 'table' not in list(_additional_kwargs_dict.keys())
        assert 'seed' not in list(_additional_kwargs_dict.keys())

    def populate_mock(self, halocat, **kwargs):
        """
        Method used to populate a simulation
        with a Monte Carlo realization of a model.

        After calling this method, the model instance
        will have a new ``mock`` attribute.
        You can then access the galaxy population via
        ``model.mock.galaxy_table``, an Astropy `~astropy.table.Table`.

        See :ref:`hod_mock_factory_source_notes`
        for an in-depth tutorial on the mock-making algorithm.

        Parameters
        ----------
        halocat : object
            Either an instance of `~halotools.sim_manager.CachedHaloCatalog`
            or `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        Num_ptcl_requirement : int, optional
            Requirement on the number of dark matter particles in the halo.
            The column defined by the ``halo_mass_column_key`` string will have a cut placed on it:
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory.
            Default value is set in `~halotools.sim_defaults.Num_ptcl_requirement`.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        halo_mass_column_key : string, optional
            This string must be a column of the input halo catalog.
            The column defined by this string will have a cut placed on it:
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory.
            Default is 'halo_mvir'.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        masking_function : function, optional
            Function object used to place a mask on the halo table prior to
            calling the mock generating functions. Calling signature of the
            function should be to accept a single positional argument storing
            a table, and returning a boolean numpy array that will be used
            as a fancy indexing mask. All masked halos will be ignored during
            mock population. Default is None.

        enforce_PBC : bool, optional
            If set to True, after galaxy positions are assigned the
            `model_helpers.enforce_periodicity_of_box` will re-map
            satellite galaxies whose positions spilled over the edge
            of the periodic box. Default is True. This variable should only
            ever be set to False when using the ``masking_function`` to
            populate a specific spatial subvolume, as in that case PBCs
            no longer apply.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        Notes
        -----
        Note the difference between the
        `halotools.empirical_models.HodMockFactory.populate` method and the
        closely related method
        `halotools.empirical_models.HodModelFactory.populate_mock`.
        The `~halotools.empirical_models.HodModelFactory.populate_mock` method
        is bound to a composite model instance and is called the *first* time
        a composite model is used to generate a mock. Calling the
        `~halotools.empirical_models.HodModelFactory.populate_mock` method creates
        the `~halotools.empirical_models.HodMockFactory` instance and binds it to
        composite model. From then on, if you want to *repopulate* a new Universe
        with the same composite model, you should instead call the
        `~halotools.empirical_models.HodMockFactory.populate` method
        bound to ``model.mock``. The reason for this distinction is that
        calling `~halotools.empirical_models.HodModelFactory.populate_mock`
        triggers a large number of relatively expensive pre-processing steps
        and self-consistency checks that need only be carried out once.
        See the Examples section below for an explicit demonstration.

        In particular, if you are running an MCMC type analysis,
        you will choose your halo catalog and completeness cuts, and call
        `populate_mock` with the appropriate arguments. Thereafter, you can
        explore parameter space by changing the values stored in the
        ``param_dict`` dictionary attached to the model, and then calling the
        `~halotools.empirical_models.MockFactory.populate` method
        bound to ``model.mock``. Any changes to the ``param_dict`` of the
        model will automatically propagate into the behavior of
        the `~halotools.empirical_models.MockFactory.populate` method.

        Examples
        ----------
        Here we'll use a pre-built model to demonstrate basic usage.
        The syntax shown below is the same for all composite models,
        whether they are pre-built by Halotools or built by you with `HodModelFactory`.

        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model_instance = PrebuiltHodModelFactory('zheng07')

        Here we will use a fake simulation, but you can populate mocks
        using any instance of `~halotools.sim_manager.CachedHaloCatalog` or
        `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> model_instance.populate_mock(halocat)

        Your ``model_instance`` now has a ``mock`` attribute bound to it.
        You can call the `~halotools.empirical_models.HodMockFactory.populate`
        method bound to the ``mock``, which will repopulate the halo catalog
        with a new Monte Carlo realization of the model.

        >>> model_instance.mock.populate()

        If you want to change the behavior of your model, just change the
        values stored in the ``param_dict``. Differences in the parameter values
        will change the behavior of the mock-population.

        >>> model_instance.param_dict['logMmin'] = 12.1
        >>> model_instance.mock.populate()

        See also
        --------
        :ref:`hod_mock_factory_source_notes`

        """
        ModelFactory.populate_mock(self, halocat, **kwargs)


##########################################
