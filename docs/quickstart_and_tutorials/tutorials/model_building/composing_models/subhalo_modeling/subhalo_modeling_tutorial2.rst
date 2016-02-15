.. _subhalo_modeling_tutorial2:

****************************************************************
Example 2: A subhalo-based model with additional features
****************************************************************

.. currentmodule:: halotools.empirical_models

In this section of the :ref:`subhalo_modeling_tutorial0`, 
we'll build a composite model that is not part of the Halotools code base 
by composing together a customized collection of Halotools-provided features. 
Before reading on, be sure you have read and understood 
:ref:`subhalo_modeling_tutorial1`. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/subhalo_modeling/subhalo_modeling_tutorial2.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 

Overview of the new model
=============================

The model we'll build will be based on the ``behroozi10`` prebuilt model, 
but we'll add an additional component model that governs whether or not 
our galaxies are quiescent or star-forming. 
The new component model we'll use is `HaloMassInterpolQuenching`. 
In this model, galaxies are assigned a boolean designation as to whether or 
not they are quiescent. Briefly, the way the model works is that you specify 
what the quiescent fraction is at a set of input control points in halo mass, 
and the model interpolates between these control points to calculate the 
quiescent fraction at any mass. See the `HaloMassInterpolQuenching` docstring for details. 

.. _subhalo_model_additional_feature_source_code:

Source code for a subhalo-based model with a new feature
=======================================================================================


>>> from halotools.empirical_models import SubhaloModelFactory
>>> from halotools.empirical_models import Behroozi10SmHm
>>> sm_model =  Behroozi10SmHm(redshift = 0)
>>> from halotools.empirical_models import HaloMassInterpolQuenching
>>> quenching = HaloMassInterpolQuenching('halo_mvir_host_halo', [1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9])
>>> behroozi10_with_quenching = SubhaloModelFactory(stellar_mass = sm_model, sfr = quenching)

Comments 
==========
First note how similar this code is to the code in :ref:`subhalo_factory_tutorial_behroozi10_source_code`. 
In fact, it is identical except for the initializing of the ``quenching``  
component model, and the additional keyword passing it to the factory. So this model 
will behave in the exact same way as the ``behroozi10`` model, except 

    1. this model has additional methods for computing the mean quiescent fraction as a function of halo mass, 
    2. when you use this model to populate mocks, the resulting ``galaxy_table`` will have a ``quiescent`` column providing a boolean designation for whether or not each mock galaxy is quiescent. 

Thus you could use this model to make predictions, for example, for the clustering of red and blue galaxy populations. Or, alternatively, you could fit the parameters of this model to observational measurements of the clustering of red/blue galaxies. 

.. _baseline_model_instance_mechanism_subhalo_model_building: 

Model-building syntax candy: the ``baseline_model_instance`` mechanism
=========================================================================

The `SubhaloModelFactory` comes with a convenient feature that makes it easier to 
add new features to existing models. By passing in a ``baseline_model_instance`` keyword argument, 
you can automatically inherit all the features of the model bound to that keyword, plus 
whatever additional arguments you may also pass to the factory. For example:

>>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
>>> ordinary_behroozi10_model = PrebuiltSubhaloModelFactory('behroozi10')

>>> from halotools.empirical_models import HaloMassInterpolQuenching
>>> quenching = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9])
>>> from halotools.empirical_models import SubhaloModelFactory
>>> behroozi10_with_quenching = SubhaloModelFactory(baseline_model_instance = ordinary_behroozi10_model, sfr = quenching)

The ``behroozi10_with_quenching`` composite model produced by the code above 
is identical in every respect to the composite model built in the 
:ref:`subhalo_model_additional_feature_source_code` section. 

The ``baseline_model_instance`` feature is designed to make it easy to study the 
effects of swapping in and out individual components without having to build a new 
model from scratch. This feature is made possible by the fact that all instances 
of Halotools composite models carry with them the instructions from which they were originally built. 
So by passing in an instance of a composite model to the `SubhaloModelFactory` via 
the ``baseline_model_instance`` keyword, the composite model instance is able to inform 
the factory of how to build a new instance of itself. 

This tutorial continues with :ref:`subhalo_modeling_tutorial3`. 




