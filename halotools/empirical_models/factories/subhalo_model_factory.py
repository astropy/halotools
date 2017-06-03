"""
Module storing the
`~halotools.empirical_models.SubhaloModelFactory` class
that governs how all subhalo-based models are built.
"""

from copy import copy
from warnings import warn
import collections

from ..factories import ModelFactory, SubhaloMockFactory

from .. import model_helpers

from ...sim_manager import sim_defaults
from ...custom_exceptions import HalotoolsError

__all__ = ['SubhaloModelFactory']
__author__ = ['Andrew Hearin']


class SubhaloModelFactory(ModelFactory):
    """ Class used to build models of the galaxy-halo connection
    in which galaxies live at the centers of subhalos.

    See :ref:`subhalo_modeling_tutorial0` for an in-depth description
    of how to build subhalo-based models, demonstrated by a
    sequence of increasingly complex examples.
    If you do not wish to build your own model but want to use one
    provided by Halotools,
    instead see `~halotools.empirical_models.PrebuiltSubhaloModelFactory`.

    All subhalo-based composite models can directly populate catalogs of dark matter halos.
    For an in-depth description of how Halotools implements this mock-generation, see
    :ref:`subhalo_mock_factory_source_notes`.

    The arguments passed to the `SubhaloModelFactory` constructor determine
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
    `~halotools.empirical_models.SubhaloModelFactory.populate_mock` method,
    as shown in the example below.

    """

    def __init__(self, **kwargs):
        """
        Parameters
        ------------------------------------
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

        galaxy_selection_func : function object, optional
            Function object that imposes a cut on the mock galaxies.
            Function should take a length-k Astropy table as a single positional argument,
            and return a length-k numpy boolean array that will be
            treated as a mask over the rows of the table. If not None,
            the mask defined by ``galaxy_selection_func`` will be applied to the
            ``galaxy_table`` after the table is generated by the `populate_mock` method.
            Default is None.

        halo_selection_func : function object, optional
            Function object used to place a cut on the input ``table``.
            If the ``halo_selection_func`` keyword argument is passed,
            the input to the function must be a single positional argument storing a
            length-N structured numpy array or Astropy table;
            the function output must be a length-N boolean array that will be used as a mask.
            Halos that are masked will be entirely neglected during mock population.

        Examples
        ------------------------------------
        As described above, there are two different ways to build models using the
        `SubhaloModelFactory`. Here we give demonstrations of each in turn.

        In the first example we'll show how to build a model from scratch using
        the ``model_features`` option.
        We'll build a composite model from two component models: one modeling stellar mass,
        one modeling star formation rate designation. We will use the
        `~halotools.empirical_models.Behroozi10SmHm` class to model stellar mass,
        and the `~halotools.empirical_models.BinaryGalpropInterpolModel` class to model
        whether galaxies are quiescent or star-forming. See the docstrings of these
        classes for more information about their behavior.

        >>> from halotools.empirical_models import Behroozi10SmHm
        >>> stellar_mass_model = Behroozi10SmHm(redshift = 0.5)

        >>> from halotools.empirical_models import BinaryGalpropInterpolModel
        >>> sfr_model = BinaryGalpropInterpolModel(galprop_name = 'quiescent_designation', galprop_abscissa = [12, 15], galprop_ordinates = [0.25, 0.75])

        At this point we have two component model instances, ``stellar_mass_model`` and
        ``sfr_model``. The following call to the factory uses the ``model_features`` option
        described above:

        >>> model_instance = SubhaloModelFactory(stellar_mass = stellar_mass_model, sfr = sfr_model)

        The feature names we have chosen are 'stellar_mass' and 'sfr', and to each feature
        we have attached a component model instance.

        In this particular example the assignment of stellar mass and SFR-designation
        are entirely independent, and so no other arguments are necessary. However, if you are
        building a model in which one or more of your components has explicit dependence on
        some other feature, then you can use the ``model_feature_calling_sequence`` argument;
        this is a list of the feature names whose order determines the sequence in which
        the components will be called during mock population:

        >>> model_instance = SubhaloModelFactory(stellar_mass = stellar_mass_model, sfr = sfr_model, model_feature_calling_sequence = ['stellar_mass', 'sfr'])

        For more details about this optional argument,
        see :ref:`model_feature_calling_sequence_mechanism`.

        Whatever features your composite model has, you can use the `~SubhaloModelFactory.populate_mock` method
        to create Monte Carlo realization of the model by populating any dark matter halo catalog
        in your cache directory:

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0.5) # doctest: +SKIP
        >>> model_instance.populate_mock(halocat) # doctest: +SKIP

        Your ``model_instance`` now has a ``mock`` attribute storing a synthetic galaxy
        population. See the `~SubhaloModelFactory.populate_mock` docstring for details.

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
        the `Moster13SmHm` class to model stellar mass.

        >>> from halotools.empirical_models import Moster13SmHm
        >>> moster_model = Moster13SmHm(redshift = 0.5)
        >>> new_model_instance = SubhaloModelFactory(stellar_mass = moster_model, baseline_model_instance = model_instance)

        The ``model_feature_calling_sequence`` works in the same way as it did in Example 1.

        >>> new_model_instance = SubhaloModelFactory(stellar_mass = moster_model, baseline_model_instance = model_instance, model_feature_calling_sequence = ['stellar_mass', 'sfr'])

        See also
        ---------
        :ref:`subhalo_model_factory_source_notes`

        :ref:`subhalo_mock_factory_source_notes`

        """

        input_model_dictionary, supplementary_kwargs = (
            self._parse_constructor_kwargs(**kwargs)
            )

        super(SubhaloModelFactory, self).__init__(input_model_dictionary, **supplementary_kwargs)

        self.mock_factory = SubhaloMockFactory
        self.model_factory = SubhaloModelFactory

        self._model_feature_calling_sequence = (
            self.build_model_feature_calling_sequence(supplementary_kwargs))

        self.model_dictionary = collections.OrderedDict()
        for key in self._model_feature_calling_sequence:
            self.model_dictionary[key] = copy(self._input_model_dictionary[key])

        # Build up and bind several lists from the component models
        self.build_prim_sec_haloprop_list()
        self.build_publication_list()
        self.build_dtype_list()
        self.set_warning_suppressions()
        self.set_inherited_methods()
        self.set_model_redshift()
        self.build_init_param_dict()

        # Create a set of bound methods with specific names
        # that will be called by the mock factory
        self.set_primary_behaviors()
        self.set_calling_sequence()
        self._test_dictionary_consistency()

    def _parse_constructor_kwargs(self, **kwargs):
        """ Method used to parse the arguments passed to
        the constructor into a model dictionary and supplementary arguments.

        `parse_constructor_kwargs` examines the keyword arguments passed to `__init__`,
        and identifies the possible presence of ``galaxy_selection_func``, ``halo_selection_func`` and
        ``model_feature_calling_sequence``; all other keyword arguments will be treated as
        component models, and it is enforced that the values bound to all such arguments
        at the very least have a ``_methods_to_inherit`` attribute.

        Parameters
        -----------
        **kwargs : optional keyword arguments
            keywords will be interpreted as the ``feature name``; values must be instances of
            Halotools component models

        Returns
        --------
        input_model_dictionary : dict
            Model dictionary defining the composite model.

        supplementary_kwargs : dict
            Dictionary of any possible remaining keyword arguments passed to the `__init__` constructor
            that are not part of the composite model dictionary, e.g., ``model_feature_calling_sequence``.

        See also
        ---------
        :ref:`subhalo_model_factory_parsing_kwargs`
        """
        if len(kwargs) == 0:
            msg = ("You did not pass any model features to the factory")
            raise HalotoolsError(msg)

        if 'baseline_model_instance' in kwargs:
            baseline_model_dictionary = kwargs['baseline_model_instance'].model_dictionary
            input_model_dictionary = copy(kwargs)
            del input_model_dictionary['baseline_model_instance']

            # First parse the supplementary keyword arguments,
            # such as 'model_feature_calling_sequence',
            # from the keywords that are bound to component model instances
            possible_supplementary_kwargs = ('galaxy_selection_func',
                'halo_selection_func', 'model_feature_calling_sequence')
            supplementary_kwargs = {}
            for key in possible_supplementary_kwargs:
                try:
                    supplementary_kwargs[key] = copy(input_model_dictionary[key])
                    del input_model_dictionary[key]
                except KeyError:
                    pass
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
            # such as 'stellar_mass'

            possible_supplementary_kwargs = ('galaxy_selection_func',
                'halo_selection_func', 'model_feature_calling_sequence')

            supplementary_kwargs = {}
            for key in possible_supplementary_kwargs:
                try:
                    supplementary_kwargs[key] = copy(input_model_dictionary[key])
                    del input_model_dictionary[key]
                except KeyError:
                    pass

            if 'model_feature_calling_sequence' not in supplementary_kwargs:
                supplementary_kwargs['model_feature_calling_sequence'] = None

            self._enforce_component_model_format(input_model_dictionary)
            return input_model_dictionary, supplementary_kwargs

    def _enforce_component_model_format(self, candidate_model_dictionary):
        """ Private method to ensure that the input model dictionary is properly formatted.
        """
        msg_preface = ("\nYou passed the following keyword argument "
            "to the SubhaloModelFactory: ``%s``\n")
        msg_conclusion = ("See the `~halotools.empirical_models.SubhaloModelFactory` "
            "docstring for further details.\n")

        for feature_key, component_model in candidate_model_dictionary.items():
            cl = component_model.__class__
            clname = cl.__name__

            if isinstance(component_model, cl):
                pass
            elif issubclass(component_model, cl):
                msg = (msg_preface + "Instead of binding an instance of ``" + clname +
                    "`` to this keyword,\n"
                    "instead you bound the ``"+clname+"`` itself.\n"
                    "The structure of Halotools model dictionaries is strictly to accept \n"
                    "component model instances, not component model classes. \n" + msg_conclusion)
                raise HalotoolsError(msg % feature_key)

    def build_model_feature_calling_sequence(self, supplementary_kwargs):
        """ Method uses the ``model_feature_calling_sequence`` passed to __init__, if available.
        If no such argument was passed, the method chooses a *mostly* random order for the calling sequence,
        excepting only for cases where either there is a feature named ``stellar_mass`` or ``luminosity``,
        which are always called first in the absence of explicit instructions to the contrary.

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
                    "that does not appear in the keyword arguments you passed to the SubhaloModelFactory.\n"
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

            if 'stellar_mass' in self._input_model_dictionary:
                model_feature_calling_sequence.append('stellar_mass')
                remaining_keys = [key for key in self._input_model_dictionary if key != 'stellar_mass']
                model_feature_calling_sequence.extend(remaining_keys)
            elif 'luminosity' in self._input_model_dictionary:
                model_feature_calling_sequence.append('luminosity')
                remaining_keys = [key for key in self._input_model_dictionary if key != 'luminosity']
                model_feature_calling_sequence.extend(remaining_keys)
            else:
                model_feature_calling_sequence = list(self._input_model_dictionary.keys())

        ########################
        # Now conversely require that all remaining __init__ constructor keyword arguments
        # appear in the model_feature_calling_sequence
        for constructor_kwarg in self._input_model_dictionary:
            try:
                assert constructor_kwarg in model_feature_calling_sequence
            except AssertionError:
                msg = ("\nYou passed ``%s`` as a keyword argument to the SubhaloModelFactory constructor.\n"
                    "This keyword argument does not appear in your input ``model_feature_calling_sequence``\n"
                    "and is otherwise not recognized.\n")
                raise HalotoolsError(msg % constructor_kwarg)
        ########################

        return model_feature_calling_sequence

    def set_inherited_methods(self):
        """ Function determines which component model methods are inherited by the composite model.

        Each component model *should* have a ``_mock_generation_calling_sequence`` attribute
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

        _method_repetition_check = []
        _attrs_repetition_check = []

        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

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

            missing_methods = (
                set(mock_making_methods) -
                set(inherited_methods).intersection(set(mock_making_methods)))
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
            list(collections.Counter(_method_repetition_check).items()) if count > 1]
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
            list(collections.Counter(_attrs_repetition_check).items()) if count > 1]
            )
        if repeated_attr_list != []:
            example_repeated_attr = repeated_attr_list[0]
            warn(repeated_attr_msg % example_repeated_attr)

    def set_primary_behaviors(self, **kwargs):
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
        `set_primary_behaviors` just creates a symbolic link to those external behaviors.

        See also
        ---------
        :ref:`subhalo_model_factory_inheriting_behaviors`
        """

        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            for methodname in component_model._methods_to_inherit:
                new_method_name = methodname
                new_method_behavior = self.update_param_dict_decorator(
                    component_model, methodname)
                setattr(self, new_method_name, new_method_behavior)

                docstring = getattr(component_model, methodname).__doc__
                getattr(self, new_method_name).__doc__ = docstring

            attrs_to_inherit = list(set(
                component_model._attrs_to_inherit))
            for attrname in attrs_to_inherit:
                new_attr_name = attrname
                attr = getattr(component_model, attrname)
                setattr(self, new_attr_name, attr)

    def update_param_dict_decorator(self, component_model, func_name):
        """
        Decorator used to propagate any possible changes in the composite model param_dict
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

        # The model dictionary is an OrderedDict that is already appropriately structured
        feature_sequence = list(self.model_dictionary.keys())

        ###############
        # Loop over feature_sequence and successively append each component model's
        # calling sequence to the composite model calling sequence
        for feature in feature_sequence:
            component_model = self.model_dictionary[feature]
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

    def set_model_redshift(self):
        """
        """

        zlist = list(model.redshift for model in list(self.model_dictionary.values())
            if hasattr(model, 'redshift'))

        if len(set(zlist)) == 0:
            self.redshift = sim_defaults.default_redshift
        elif len(set(zlist)) == 1:
            self.redshift = float(zlist[0])
        else:
            msg = ("Inconsistency between the redshifts of the component models:\n\n")
            for modelname, model in self.model_dictionary.items():
                clname = model.__class__.__name__
                if hasattr(model, 'redshift'):
                    zs = str(model.redshift)
                    msg += ("For modelname = ``" + modelname + "``, the " +
                        clname+" instance has redshift = " + zs + "\n")
            raise HalotoolsError(msg)

    def build_prim_sec_haloprop_list(self):
        """ Method builds the ``_haloprop_list`` of strings.

        This list stores the names of all halo catalog columns
        that appear as either ``prim_haloprop_key`` or ``sec_haloprop_key`` of any component model.
        For all strings appearing in ``_haloprop_list``, the mock ``galaxy_table`` will have
        a corresponding column storing the halo property inherited by the mock galaxy.
        """
        haloprop_list = []
        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            if hasattr(component_model, 'prim_haloprop_key'):
                haloprop_list.append(component_model.prim_haloprop_key)
            if hasattr(component_model, 'sec_haloprop_key'):
                haloprop_list.append(component_model.sec_haloprop_key)
            if hasattr(component_model, 'list_of_haloprops_needed'):
                haloprop_list.extend(component_model.list_of_haloprops_needed)

        self._haloprop_list = list(set(haloprop_list))

    def build_publication_list(self):
        """ Method collects together all publications from each of the component models.
        """
        pub_list = []
        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            try:
                pubs = component_model.publications
                if type(pubs) in [str, str]:
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
        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            if hasattr(component_model, '_suppress_repeated_param_warning'):
                self._suppress_repeated_param_warning += component_model._suppress_repeated_param_warning

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

        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            if not hasattr(component_model, 'param_dict'):
                component_model.param_dict = {}

            intersection = set(self.param_dict) & set(component_model.param_dict)

            if intersection != set():
                for key in intersection:
                    if suppress_warning is False:
                        warn(msg % key)

            for key, value in component_model.param_dict.items():
                self.param_dict[key] = value

        self._init_param_dict = copy(self.param_dict)

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
        # Loop over all component features in the composite model
        for feature, component_model in self.model_dictionary.items():

            # Column dtypes to add to mock galaxy_table
            if hasattr(component_model, '_galprop_dtypes_to_allocate'):
                dtype_list.append(component_model._galprop_dtypes_to_allocate)

        self._galprop_dtypes_to_allocate = model_helpers.create_composite_dtype(dtype_list)

    def restore_init_param_dict(self):
        """ Reset all values of the current ``param_dict`` to the values
        the class was instantiated with.

        Primary behaviors are reset as well, as this is how the
        inherited behaviors get bound to the values in ``param_dict``.

        See also
        ---------
        :ref:`param_dict_mechanism`
        """
        self.param_dict = self._init_param_dict
        self.set_primary_behaviors()
        self.set_calling_sequence()

    def populate_mock(self, halocat, masking_function=None, **kwargs):
        """
        Method used to populate a simulation
        with a Monte Carlo realization of a model.

        After calling this method, the model instance
        will have a new ``mock`` attribute.
        You can then access the galaxy population via
        ``model.mock.galaxy_table``, an Astropy `~astropy.table.Table`.

        See :ref:`subhalo_mock_factory_source_notes`
        for an in-depth tutorial on the mock-making algorithm.

        Parameters
        ----------
        halocat : object
            Either an instance of `~halotools.sim_manager.CachedHaloCatalog`
            or `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        masking_function : function, optional
            Function object used to place a mask on the halo table prior to
            calling the mock generating functions. Calling signature of the
            function should be to accept a single positional argument storing
            a table, and returning a boolean numpy array that will be used
            as a fancy indexing mask. All masked halos will be ignored during
            mock population. Default is None.

        Notes
        -----
        Note the difference between the
        `halotools.empirical_models.SubhaloMockFactory.populate` method and the
        closely related method
        `halotools.empirical_models.SubhaloModelFactory.populate_mock`.
        The `~halotools.empirical_models.SubhaloModelFactory.populate_mock` method
        is bound to a composite model instance and is called the *first* time
        a composite model is used to generate a mock. Calling the
        `~halotools.empirical_models.SubhaloModelFactory.populate_mock` method creates
        the `~halotools.empirical_models.SubhaloMockFactory` instance and binds it to
        composite model. From then on, if you want to *repopulate* a new Universe
        with the same composite model, you should instead call the
        `~halotools.empirical_models.SubhaloMockFactory.populate` method
        bound to ``model.mock``. The reason for this distinction is that
        calling `~halotools.empirical_models.SubhaloModelFactory.populate_mock`
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
        whether they are pre-built by Halotools or built by you with `SubhaloModelFactory`.

        >>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
        >>> model_instance = PrebuiltSubhaloModelFactory('behroozi10')

        Here we will use a fake simulation, but you can populate mocks
        using any instance of `~halotools.sim_manager.CachedHaloCatalog` or
        `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> model_instance.populate_mock(halocat)

        Your ``model_instance`` now has a ``mock`` attribute bound to it.
        You can call the `~halotools.empirical_models.SubhaloMockFactory.populate`
        method bound to the ``mock``, which will repopulate the halo catalog
        with a new Monte Carlo realization of the model.

        >>> model_instance.mock.populate()

        If you want to change the behavior of your model, just change the
        values stored in the ``param_dict``. Differences in the parameter values
        will change the behavior of the mock-population.

        >>> model_instance.param_dict['scatter_model_param1'] = 0.25
        >>> model_instance.mock.populate()

        See also
        -----------
        :ref:`subhalo_mock_factory_source_notes`

        """
        if masking_function is not None:
            ModelFactory.populate_mock(self, halocat, masking_function=masking_function, **kwargs)
        else:
            ModelFactory.populate_mock(self, halocat, **kwargs)

    def _test_dictionary_consistency(self):
        """
        """
        for component_model in list(self.model_dictionary.values()):
            try:
                assert hasattr(component_model, '_methods_to_inherit')
                for methodname in component_model._methods_to_inherit:
                    assert hasattr(component_model, methodname)
            except AssertionError:
                clname = component_model.__class__.__name__
                msg = ("You bound an instance of the ``"+clname+"`` to this keyword,\n"
                    "but the instance does not have "
                    "a properly defined ``_methods_to_inherit`` attribute.\n"
                    "At a minimum, all component models must have this attribute, \n"
                    "even if there is only an empty list bound to it.\n"
                    "Any items in this list must be names of methods bound to "
                    "the component model.\n")
                raise HalotoolsError(msg)
