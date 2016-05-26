:orphan:

.. currentmodule:: halotools.empirical_models

.. _composite_model_constructor_bookkeeping_mechanisms:

****************************************************
Composite Model Bookkeeping Mechanisms
****************************************************

.. _param_dict_mechanism:

The ``param_dict`` mechanism
================================================

The component model classes in Halotools determine the functional form of the aspect of the galaxy-halo connection being modeled, but in many cases models have a handful of parameters allows you to tune the behavior of the functions. For example, the `~halotools.empirical_models.Zheng07Cens` class dictates that the average number of central galaxies as a function of halo mass, :math:`\langle N_{\rm cen}\vert M_{\rm halo}\rangle`, is governed by an ``erf`` function, but the speed of the transition of the ``erf`` function between 0 and 1 can be varied by changing the :math:`\sigma_{\log M}` parameter. 

In all such cases, parameters such as :math:`\sigma_{\log M}` are elements of ``param_dict``, a python dictionary bound to the component model instance. By changing the values bound to the parameters in the ``param_dict``, you change the behavior of the model. 

Propagating ``param_dict`` from component to composite 
--------------------------------------------------------
While creating a composite model from a set of component models, the factory classes `~SubhaloModelFactory` and `~HodModelFactory` collect every parameter that appears in each component model ``param_dict``, and create a new composite ``param_dict`` that is bound to the composite model instance. The way that composite model methods are written, in order to change the behavior of the composite model all you need to do is change the values of the parameters in the ``param_dict`` bound to the composite model and the changes propagate down to the component model defining the behavior. 

In most cases this propagation process is unambiguous and straightforwardly accomplished with :ref:`update_param_dict_decorator_mechanism`. However, if two or more component models have a parameter with the exact same name, then care is required. 

As an example, consider the composite model dictionary built by the `~halotools.empirical_models.leauthaud11_model_dictionary` function. In this composite model, there are two populations of galaxies, *centrals* and *satellites*, whose occupation statistics are governed by `~halotools.empirical_models.Leauthaud11Cens` and `~halotools.empirical_models.Leauthaud11Sats`, respectively. Both of these classes derive much of their behavior from the underlying stellar-to-halo-mass relation of Behroozi et al. (2010), and so all the parameters in the ``param_dict`` of `~halotools.empirical_models.Behroozi10SmHm` appear in both the ``param_dict`` of `~halotools.empirical_models.Leauthaud11Cens` and the ``param_dict`` of `~halotools.empirical_models.Leauthaud11Sats`. 

In this example, the repeated appearance of the stellar-to-halo-mass parameters is harmless because these these really are the same parameters that just so happen to appear twice. But since Halotools users are free to define their own model components and compose any arbitrary collection of components together, it is possible that the same name could have been inadvertently given to parameters in different components controlling entirely distinct behavior. In such a case, when that parameter is modified in the composite model ``param_dict`` it is ambiguous how to propagate the change down in to the appropriate component model. 

Suppressing harmless warnings 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To protect against this ambiguity, whenever a repeated parameter is detected during the building of the composite model ``param_dict``, a warning is issued to the user. It is up to the user to determine whether the repetition is harmless or if one of the component model parameter names needs to be changed to disambiguate. As the appearance of such a warning can be annoying for commonly-used models in which the repetition is harmless, it is possible to suppress this warning by creating a ``_suppress_repeated_param_warning`` attribute to *any* of the components in the composite model, and setting this attribute to ``True``. You can see this mechanism at work in the source code of the `~halotools.empirical_models.leauthaud11_model_dictionary` function. 


.. _new_haloprop_func_dict_mechanism:

The ``new_haloprop_func_dict`` mechanism
============================================================

The basic job of any component model is to provide some mapping between a halo catalog and 
some property (or set of properties) of the galaxies that live in those halos. There are many 
cases where the underlying halo property that is the independent variable in the mapping 
is not a pre-existing column in your halo catalog. For example, assigning positions to satellite 
galaxies may depend on an analytical model for the NFW concentration-mass relation. If this particular 
definition of the concentration is not already in the halo catalog, this quantity would need to be 
computed for every halo before the satellite positions could be assigned. 

There are several possible solutions to this problem. First, the composite model could simply compute 
the desired halo property on-the-fly as part of the galaxy property assignment. However, if the 
calculation is expensive then this needlessly adds runtime to mock-population with any composite model that uses this component. Second, you could always add the desired column to the halo table and then over-write the existing halo catalog data file with an updated one that includes the new column. 
However, for the sake of reproducibility it is best to minimize the number of times 
a halo catalog is over-written, as keeping track of each over-write quickly becomes a headache 
and mistakes in that bookkeeping can lead to hidden buggy behavior.  

The ``new_haloprop_func_dict`` mechanism helps address this problem. When model factories 
build composite models, each component model is examined for the possible presence of a 
``new_haloprop_func_dict`` attribute, to which a python dictionary must be bound. 
The keys of this dictionary serve as the names of the new columns that will be added to the halo catalog 
in a pre-processing phase of the mock-population algorithm. 
The values bound to these keys are python function objects; each function object must accept a length-*Nhalos*
Astropy `~astropy.table.Table` or Numpy structured array as input, and it must return a length-*Nhalos* array as output; the returned array will be the data in the newly created column of the halo catalog. 

To take advantage of this mechanism in your component model, the only thing you need to do is create a 
``new_haloprop_func_dict`` attribute somewhere in the `__init__` constructor of your component model, 
and make sure that the dictionary bound to this attribute conforms to the above specifications. After doing this, you can safely assume that the halo catalog column needed by your component model will be in any halo catalog used to populate mock galaxies with a composite model using your component. 


.. _galprop_dtypes_to_allocate_mechanism:

The ``galprop_dtypes_to_allocate`` mechanism
============================================================

Whenever a component model is used during mock population, the mock factory passes a ``table`` keyword 
argument to the methods of the component. It is important that the table passed to the function has the necessary columns assumed by the function. 

Every component model assigns some property or set of properties to the mock population of galaxies. In mock population, the synthetic galaxy population is stored in the ``galaxy_table`` bound to the mock object. The ``galaxy_table`` is an Astropy `~astropy.table.Table` object, with columns storing every galaxy property assigned by the composite model. The ``_galprop_dtypes_to_allocate`` mechanism is responsible for creating the necessary columns of the ``galaxy_table`` and making sure they are appropriately formatted. 

If you are writing your own model component of any kind, the model factories require that instances of your model have a ``_galprop_dtypes_to_allocate`` attribute. You can meet this specification by assigning any `numpy.dtype` object to the ``_galprop_dtypes_to_allocate`` attribute during the `__init__` constructor of your componenent model (even if the dtype is empty). See :ref:`composing_new_models` for many examples. 

.. _model_feature_calling_sequence_mechanism:

The ``model_feature_calling_sequence`` mechanism
======================================================================

When the mock factories create a synthetic galaxy population, a sequence of methods of the composite model are called in the order determined by the ``_mock_generation_calling_sequence`` list attribute bound to the *composite* model. For subhalo-based models, this list is determined by `SubhaloModelFactory.set_calling_sequence`, whereas for HOD-style models this list is determined by `HodModelFactory.set_calling_sequence`. 

As described in :ref:`mock_generation_calling_sequence_mechanism`, each *component* model also has a ``_mock_generation_calling_sequence`` attribute. The composite model sequence is built up as a succession 
of the component model sequences. The sequential ordering of component models in this succession is determined by the ``_model_feature_calling_sequence`` attribute, which is set by the `build_model_feature_calling_sequence` factory method. Thus the composite model ``_mock_generation_calling_sequence`` is determined according to the following schematic:

.. code-block:: python

	composite_model._mock_generation_calling_sequence = []
	for component_model_name in composite_model._model_feature_calling_sequence:
		component_model = composite_model.model_dictionary[component_model_name]
		for method_name in component_model._mock_generation_calling_sequence:
			composite_model._mock_generation_calling_sequence.append(method_name)

Thus each component model's methods are always called one right after the other. The order in which each component model is called upon is determined by the ``_model_feature_calling_sequence`` attribute. The user is free to explicitly specify this sequence via the model_feature_calling_sequence keyword argument passed to the factory constructor. This may be useful for cases where the model for one galaxy property has explicit dependende on another galaxy property defined in an independent model component. If the model_feature_calling_sequence keyword is not passed, the order in which the component models are called should be assumed to be random. 

.. _mock_generation_calling_sequence_mechanism:

The ``mock_generation_calling_sequence`` mechanism
======================================================================

Each component model has a ``_mock_generation_calling_sequence`` attribute storing a list of strings. Each string is the name of a method bound to the component model instance. The order in which these names appear determines the order in which the methods will be called during mock population. This mechanism works together with :ref:`model_feature_calling_sequence_mechanism` to determine the entire sequence of functions that are called when populating a mock. 


.. _update_param_dict_decorator_mechanism:

The ``update_param_dict_decorator`` mechanism
=================================================

As described in :ref:`param_dict_mechanism`, the composite model ``param_dict`` is simply a collection of the parameters in the ``param_dict`` of all the component models. While this collection process is simple, it creates the following problem. The component and composite ``param_dict`` are separate dictionaries, and even though they share keys in common, the keys point to different locations in memory. So if the user decides to change the value bound to a key in the ``param_dict`` of the composite model, this change does nothing at all to the value bound to the corresponding key the component model. And yet, the behavior is *entirley* governed by the component model, so unless some action is taken to propagate the change from the composite ``param_dict`` to the component ``param_dict``, then the composite model will not change behavior when its ``param_dict`` is changed. 

The `ModelFactory.update_param_dict_decorator` addresses this problem. When the model factories inherit the methods of the component models, they actually inherited modified versions of the methods, where the modification comes from decorating the inherited methods with the `~ModelFactory.update_param_dict_decorator`, whose source code appears below:

.. code-block:: python

    def update_param_dict_decorator(self, component_model, func_name):

        def decorated_func(*args, **kwargs):

            # Update the param_dict as necessary
            for key in self.param_dict.keys():
                if key in component_model.param_dict:
                    component_model.param_dict[key] = self.param_dict[key]

            func = getattr(component_model, func_name)
            return func(*args, **kwargs)

        return decorated_func

The behavior of the `decorated_func` is identical in every way to the input function, except for before calling the 
input function, `decorated_func` first opens up the component model ``param_dict`` and updates any the value of any key that also appears in the composite model ``param_dict``. 

Note that this mechanism does *not* automatically and immediately propagate changes in the composite model ``param_dict`` to the component model ``param_dict``. If you manually change values in the composite model ``param_dict``, nothing happens to the component model by that action alone. The role of the `~ModelFactory.update_param_dict_decorator` is to accomplish this propagation when it counts: when you actually call the methods of the component model that the composite model actually needs. 

.. _list_of_haloprops_needed_mechanism:

The ``list_of_haloprops_needed`` mechanism 
============================================

When the `MockFactory` calls upon the component model methods, the only thing that 
gets passed to each methods is a ``table`` keyword argument. In *almost* all cases, 
the table bound to this keyword is the ``galaxy_table`` that is in the process of 
being generated (see the :ref:`galprops_assigned_before_mc_occupation` section of the 
:ref:`hod_mock_factory_source_notes` documentation page for the only exception to this rule). 

The ``galaxy_table`` differs from the ``halo_table`` in several respects. 
In subhalo-based models, they will have the same length, but in HOD-style models 
they will generally have different lengths. The ``galaxy_table`` will have columns 
associated with mock galaxy properties that the ``halo_table`` generally will not. 

For the purpose of this discussion, the most important difference is this: 
*the ``galaxy_table`` only inherits the columns of the ``halo_table`` that the 
composite model tells it to inherit.* The ``list_of_haloprops_needed`` is the 
mechanism that the composite model exploits to inform the `MockFactory` which ``halo_table`` 
columns should be inherited by the ``galaxy_table``. 

All component models have the option to define a ``list_of_haloprops_needed`` attribute, 
a list of strings of ``halo_table`` column names. The model factory collects together all these lists 
and forms their union. Any ``halo_table`` column name in this union will be inherited by 
the mock galaxy population. Component models need not necessarily define 
a ``list_of_haloprops_needed`` attribute. For example, in cases where 
multiple component models require the same halo property, only one component need 
declare a need for this property. Multiple requests of the same column is always harmless, 
but for if you ever choose to include a component model that does not include a 
``list_of_haloprops_needed`` attribute, 
the model factory will always raise a (possibly harmless) warning. 















