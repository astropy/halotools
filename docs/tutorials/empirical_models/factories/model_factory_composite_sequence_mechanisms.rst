:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _composite_model_constructor_bookkeeping_mechanisms:

****************************************************
Composite Model Constructor Bookkeeping Mechanisms
****************************************************

This tutorial remains to be written. See `Github Issue 211 <https://github.com/astropy/halotools/issues/211/>`_


.. _param_dict_mechanism:

The ``param_dict`` mechanism
================================================

The component model classes in Halotools determine the functional form of the aspect of the galaxy-halo connection being modeled, but in many cases models have a handful of parameters allows you to tune the behavior of the functions. For example, the `~halotools.empirical_models.occupation_models.zheng07_components.Zheng07Cens` class dictates that the average number of central galaxies as a function of halo mass, :math:`\langle N_{\rm cen}\vert M_{\rm halo}\rangle`, is governed by an ``erf`` function, but the speed of the transition of the ``erf`` function between 0 and 1 can be varied by changing the :math:`\sigma_{\log M}` parameter. 

In all such cases, parameters such as :math:`\sigma_{\log M}` are elements of ``param_dict``, a python dictionary bound to the component model instance. By changing the values bound to the parameters in the ``param_dict``, you change the behavior of the model. 

While creating a composite model from a set of component models, the factory classes `~SubhaloModelFactory` and `~HodModelFactory` collect every parameter that appears in each component model ``param_dict``, and create a new composite ``param_dict`` that is bound to the composite model instance. The way that composite model methods are written, in order to change the behavior of the composite model all you need to do is change the values of the parameters in the ``param_dict`` bound to the composite model and the changes propagate down to the component model defining the behavior. 

In most cases this propagation process is unambiguous and straightforwardly accomplished with the `_update_param_dict_decorator` python decorator in the factory. However, if two or more component models have a parameter with the exact same name, then care is required. 

As an example, consider the composite model dictionary built by the `~halotools.empirical_models.composite_models.hod_models.leauthaud11_model_dictionary` function. In this composite model, there are two populations of galaxies, *centrals* and *satellites*, whose occupation statistics are governed by `~halotools.empirical_models.occupation_models.Leauthaud11Cens` and `~halotools.empirical_models.occupation_models.Leauthaud11Sats`, respectively. Both of these classes derive much of their behavior from the underlying stellar-to-halo-mass relation of Behroozi et al. (2010), and so all the parameters in the ``param_dict`` of `~halotools.empirical_models.smhm_models.Behroozi10SmHm` appear in both the ``param_dict`` of `~halotools.empirical_models.occupation_models.Leauthaud11Cens` and the ``param_dict`` of `~halotools.empirical_models.occupation_models.Leauthaud11Sats`. 

In this example, the repeated appearance of the stellar-to-halo-mass parameters is harmless because these these really are the same parameters that just so happen to appear twice. But since Halotools users are free to define their own model components and compose any arbitrary collection of components together, it is possible that the same name could have been inadvertently given to parameters in different components controlling entirely distinct behavior. In such a case, when that parameter is modified in the composite model ``param_dict`` it is ambiguous how to propagate the change down in to the appropriate component model. 

To protect against this ambiguity, whenever a repeated parameter is detected during the building of the composite model ``param_dict``, a warning is issued to the user. It is up to the user to determine whether the repetition is harmless or if one of the component model parameter names needs to be changed to disambiguate. As the appearance of such a warning can be annoying for commonly-used models in which the repetition is harmless, it is possible to suppress this warning by creating a ``_suppress_repeated_param_warning`` attribute to *any* of the components in the composite model, and setting this attribute to ``True``. You can see this mechanism at work in the source code of the `~halotools.empirical_models.composite_models.hod_models.leauthaud11_model_dictionary` function. 


.. _new_haloprop_func_dict_mechanism:

The ``new haloprop func dict`` mechanism
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

The ``galprop dtypes to allocate`` mechanism
============================================================


.. _mock_generation_calling_sequence_mechanism:

The ``mock generation calling sequence`` mechanism
======================================================================

.. _model_feature_calling_sequence_mechanism:

The ``model feature calling sequence`` mechanism
======================================================================





