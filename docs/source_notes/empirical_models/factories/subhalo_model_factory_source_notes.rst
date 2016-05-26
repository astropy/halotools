:orphan:

.. currentmodule:: halotools.empirical_models

.. _subhalo_model_factory_source_notes:

****************************************************
Source code notes on `SubhaloModelFactory`
****************************************************

This section of the documentation provides detailed notes 
on the source code implementation of the `SubhaloModelFactory` class. 
The purpose of the `SubhaloModelFactory` class is to provide a flexible, standardized platform for building subhalo-based models that can directly populate simulations with mock galaxies. The goal is to make it easy to swap new modeling features in and out of the framework while maintaining a uniform syntax. This way, when you want to study one particular feature of the galaxy-halo connection, you can focus exclusively on developing that feature, leaving the factory to take care of the remaining aspects of the mock population. This tutorial describes in detail how the `~SubhaloModelFactory` accomplishes this standardization. 

Outline 
========

We will start in :ref:`subhalo_model_factory_design_overview` with a high-level 
description of how the class creates a composite model from 
a set of independently-defined features. In :ref:`subhalo_model_factory_parsing_kwargs` we describe 
how the factory's `__init__` constructor parses the large number of optional inputs into a *model dictionary*. 
In :ref:`subhalo_model_factory_bookkeeping_mechanisms` we outline the various bookkeeping devices and consistency checks that the factory does in order to 1. ensure that the input model dictionary provides sufficient and self-consistent information, and 2. place the instance into a form that can directly talk to the `~SubhaloMockFactory`. In :ref:`subhalo_model_factory_inheriting_behaviors` we cover the process by which the appropriate methods of the component models are inherited by the composite model. The syntax for using a composite model to create mock catalogs is covered in :ref:`populate_subhalo_mock_convenience_method`. 
We conclude in :ref:`subhalo_model_factory_further_reading` by pointing to sections of documentation covering related aspects such as the algorithm for using `SubhaloModelFactory` instances to populate mocks. 

.. _subhalo_model_factory_design_overview:

Overview of the factory design
=================================

The `SubhaloModelFactory` has virtually no behavior of its own; 
it should instead be thought of as a container class that collects together 
behaviors that are defined elsewhere. These behaviors are defined in 
*component models*, which are instances of Halotools classes that typically provide 
a single, specialized mapping between halos and some specific galaxy property. 
By composing these individual mappings together, 
the output of the factory is a composite model for the galaxy-halo connection in which 
any number of user-defined galaxy properties is simultaneously modeled. 

Although there are numerous options 
for the form of the arguments passed to `SubhaloModelFactory`, 
the basic input is a *model dictionary*. 
A model dictionary is just an ordinary python dictionary that stores the collection of 
component model instances whose behaviors are being unified together by the factory. 
The model dictionary contains all of the necessary information inform the `SubhaloModelFactory` 
how to build a composite model from the components. 

Each component model in a model dictionary typically has each of the following three private attributes:

    1. `_methods_to_inherit`

    2. `_galprop_dtypes_to_allocate`

    3. `_mock_generation_calling_sequence`

Each of these three attributes will be explained in detail below. Briefly, the 
`_methods_to_inherit` is a list of strings that instructs the `~SubhaloModelFactory` 
which methods in the component model should be carried over into the composite model. 
The `_galprop_dtypes_to_allocate` attribute is used to instruct the `~SubhaloMockFactory` 
of the shape and name of every Numpy array that should be allocated for every galaxy property 
assigned by the component model. The `_mock_generation_calling_sequence` specifies the sequential 
order in which the methods of the component model should be called by the composite model 
during mock population. 

Again, we will discuss these and other bookkeeping devices in more detail below. 
For now, simply observe what is accomplished by these three pieces of information. 
Each component model is effectively giving the factory the following message:
"I want you to know about the following methods, and only the following methods, and I will take care of how they will be computed: `_methods_to_inherit`; 
I need you to make sure that the when you call these methods, the following arrays that will be passed to them: `_galprop_dtypes_to_allocate`; when you use me to make a mock, I need you to call these
methods in the following sequence: `_mock_generation_calling_sequence`". In this way, not only is all the physically relevant behavior defined in the component models, but the component models themselves provide the instructions for how they should be used. 

The job of the `~SubhaloModelFactory` is simply to follow these instructions, and to ensure that mutually consistent messages are received from the set of components in the model dictionary. In the remaining sections of this tutorial, we will walk step-by-step through the tasks carried out when a new composite model is built by instantiating an instance of the `~SubhaloModelFactory` class. 

.. _subhalo_model_factory_parsing_kwargs:

Inferring a model dictionary from the constructor inputs 
===========================================================

The first thing the `__init__` constructor of `~SubhaloModelFactory` does is to 
pass all its arguments to the `~SubhaloModelFactory._parse_constructor_kwargs` method, 
which simply extracts (if present) ``galaxy_selection_func``, ``halo_selection_func`` and ``model_feature_calling_sequence`` from the arguments passed to ``__init__``; 
all remaining arguments will be interpeted as model dictionary inputs. 
For an explanation of ``galaxy_selection_func`` and ``halo_selection_func``, 
see the `~ModelFactory` docstring. 

When calling the constructor of the `~ModelFactory` super-class after parsing the inputs, 
exact copies of all arguments passed to `~SubhaloModelFactory` are bound to the instance. 
This allows all composite model instances to remember the 
exact set of instructions from which they were built. 
As we will see, this is useful because it simplifies the process of building 
alternate versions of any particular composite model instance. 

As described in :ref:`model_feature_calling_sequence_mechanism`, 
the ``model_feature_calling_sequence`` determines 
the order in which the component models will be called during mock population. This order is 
determined by the `~SubhaloModelFactory.build_model_feature_calling_sequence` method. 

Once this order is determined, the ``model_dictionary`` attribute is bound to the instance
using the appropriate order:

.. code-block:: python

    self.model_dictionary = collections.OrderedDict()
    for key in self._model_feature_calling_sequence:
        self.model_dictionary[key] = copy(self._input_model_dictionary[key])

In the next section, we will see how the ``model_dictionary`` attribute is used to create a 
number of bookkeeping mechanisms used to verify self-consistency between the model features, 
and also to facilitate communication between the composite model and the `~SubhaloMockFactory`. 

.. _subhalo_model_factory_bookkeeping_mechanisms:

Consistency checks and mock-population bookkeeping  
================================================================

After the model dictionary has been built, the `__init__` constructor 
creates a handful of lists and dictionaries and binds these to the instance 
with the following lines of code: 

.. code-block:: python

    # Build up and bind several lists from the component models
    self.build_prim_sec_haloprop_list()
    self.build_publication_list()
    self.build_new_haloprop_func_dict()
    self.build_dtype_list()
    self.set_warning_suppressions()
    self.set_model_redshift()
    self.set_inherited_methods()
    self.build_init_param_dict()

These methods examine each of the component models, perform various self-consistency 
tests, and create standardized attributes that allow the 
composite model to communicate with the `~SubhaloMockFactory` to populate mocks. 
For a description of the most important methods in this standardization process, 
see :ref:`composite_model_constructor_bookkeeping_mechanisms`. At the end of this 
sequence of function calls, the instance is prepared to inherit the behavior of 
the primary methods of the component models, which we cover in the next section. 


.. _subhalo_model_factory_inheriting_behaviors:

Inheriting behaviors from the component models
=================================================

Once all of the above lists and dictionaries of the composite model have been created, 
the `~SubhaloModelFactory` finally inherits the behaviors of the component models. 
This is done using with the `~SubhaloModelFactory.set_primary_behaviors` method. 

This is the most important function in the entire factory. Although it is only a few lines, 
it is sufficiently complicated to warrant detailed discussion. 
First, we reproduce the source below: 

.. code-block:: python

    for feature, component_model in self.model_dictionary.iteritems():

        for methodname in component_model._methods_to_inherit:

            new_method_name = methodname # line 1
            new_method_behavior = self.update_param_dict_decorator(
                component_model, methodname) # line 2
            setattr(self, new_method_name, new_method_behavior) # line 3


In this double-for loop, we iterate over every method that the composite model 
should inherit from the collection of component models. 
For each method that we inherit, line 3 binds the newly-defined method to the composite model instance. 
Line 1 chooses for the name of this newly-defined method to keep the same name 
as appears in the component model. Line 2 modifies the component model method behavior with the 
`~ModelFactory.update_param_dict_decorator` decorator. 
This modification is very important for the reasons described in :ref:`update_param_dict_decorator_mechanism`. 

Note how the use of getattr and setattr allows the component models to entirely dictate 
what is inherited by the composite model. This high-level python feature is what makes possible 
the flexibility of the model factories. 


.. _populate_subhalo_mock_convenience_method:

The ``populate_mock`` convenience method
=====================================================

No matter what the component model features are, all instances of `SubhaloModelFactory` 
can directly populate subhalo catalogs with mock galaxies 
with the `~SubhaloModelFactory.populate_mock` method. To populate the default halo catalog, 
the syntax for this is:

.. code-block:: python

    model = SubhaloModelFactory(**model_dictionary)
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog()
    model.populate_mock(halocat)

The `SubhaloModelFactory.populate_mock` method is just a 
convenience wrapper around `SubhaloMockFactory.populate` method. 

You can also populate alternative halo catalogs:

.. code-block:: python

    from halotools.sim_manager import CachedHaloCatalog
    my_halocat = CachedHaloCatalog(simname = my_simname, redshift = my_redshift)
    model.populate_mock(my_halocat)

You can use the syntax above to populate any instance of either 
`~halotools.sim_manager.CachedHaloCatalog` or `~halotools.sim_manager.UserSuppliedHaloCatalog`. 

.. _subhalo_model_factory_further_reading:

Further reading 
================

Detailed documentation on the mock-population algorithm is covered in :ref:`subhalo_mock_factory_source_notes`. 





