.. _hod_modeling_tutorial2:

****************************************************************
Example 2: An HOD-style model with additional features
****************************************************************

.. currentmodule:: halotools.empirical_models

In this section of the :ref:`hod_modeling_tutorial0`, 
we'll build a composite model that is not part of the Halotools code base 
by composing together a customized collection of Halotools-provided features. 
Before reading on, be sure you have read and understood 
:ref:`hod_modeling_tutorial1`. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


	**halotools/docs/notebooks/hod_modeling/hod_modeling_tutorial2.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 

Overview of the new model
=============================

The model we'll build will be based on the ``zheng07`` HOD, 
but we'll add an additional component model that governs whether or not 
our centrals and satellites are quiescent or star-forming. 
The new component model we'll use is `HaloMassInterpolQuenching`. 
In this model, galaxies are assigned a boolean designation as to whether or 
not they are quiescent. Briefly, the way the model works is that you specify 
what the quiescent fraction is at a set of input control points in halo mass, 
and the model interpolates between these control points to calculate the 
quiescent fraction at any mass. See the `HaloMassInterpolQuenching` docstring for details. 


.. _hod_model_additional_feature_source_code: 

Source code for an HOD model with a new feature 
=================================================

.. code:: python

	from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
	another_cens_occ_model =  Zheng07Cens()
	another_cens_prof_model = TrivialPhaseSpace()

	from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
	another_sats_occ_model =  Zheng07Sats()
	another_sats_prof_model = NFWPhaseSpace()

	from halotools.empirical_models import HaloMassInterpolQuenching
	sat_quenching = HaloMassInterpolQuenching('halo_mvir', 
		[1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9], 
		gal_type = 'satellites')
	cen_quenching = HaloMassInterpolQuenching('halo_mvir', 
		[1e12, 1e15], [0.25, 0.95], 
		gal_type = 'centrals')

	from halotools.empirical_models import HodModelFactory
	zheng07_with_quenching = HodModelFactory(
		centrals_occupation = another_cens_occ_model, 
		centrals_profile = another_cens_prof_model, 
		satellites_occupation = another_sats_occ_model, 
		satellites_profile = another_sats_prof_model, 
		centrals_quenching = cen_quenching, 
		satellites_quenching = sat_quenching
		)

Comments 
==========
First note how similar this code is to the code in :ref:`hod_factory_tutorial_zheng07_source_code`. 
In fact, it is identical except for the initializing of the ``cen_quenching`` and ``sat_quenching`` 
component models, and the additional keywords passing them to the factory. So this model 
will behave in the exact same way as the ``zheng07`` model, except 

	1. this model has additional methods for computing the mean quiescent fraction as a function of halo mass for centrals and satellites, 
	2. when you use this model to populate mocks, the resulting ``galaxy_table`` will have a ``quiescent`` column providing a boolean designation for whether or not each mock galaxy is quiescent. 

Thus you could use this model to make predictions, for example, for the clustering of red and blue galaxy populations. Or, alternatively, you could fit the parameters of this model to observational measurements of the clustering of red/blue galaxies. 


.. _baseline_model_instance_mechanism_hod_building: 

Model-building syntax candy: the ``baseline_model_instance`` mechanism
=========================================================================

The `HodModelFactory` comes with a convenient feature that makes it easier to 
add new features to existing models. By passing in a ``baseline_model_instance`` keyword argument, 
you can automatically inherit all the features of the model bound to that keyword, plus 
whatever additional arguments you may also pass to the factory. For example:

.. code:: python

	from halotools.empirical_models import PrebuiltHodModelFactory
	ordinary_zheng07_model = PrebuiltHodModelFactory('zheng07')

	from halotools.empirical_models import HaloMassInterpolQuenching
	sat_quenching = HaloMassInterpolQuenching('halo_mvir', 
		[1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9], 
		gal_type = 'satellites')
	cen_quenching = HaloMassInterpolQuenching('halo_mvir', 
		[1e12, 1e15], [0.25, 0.95], 
		gal_type = 'centrals')

	from halotools.empirical_models import HodModelFactory
	zheng07_with_quenching = HodModelFactory(
		baseline_model_instance = ordinary_zheng07_model, 
		centrals_quenching = cen_quenching, 
		satellites_quenching = sat_quenching
		)

The ``zheng07_with_quenching`` composite model produced by the code above 
is identical in every respect to the composite model built in the 
:ref:`hod_model_additional_feature_source_code` section. 

The ``baseline_model_instance`` feature is designed to make it easy to study the 
effects of swapping in and out individual components without having to build a new 
model from scratch. This feature is made possible by the fact that all instances 
of Halotools composite models carry with them the instructions from which they were originally built. 
So by passing in an instance of a composite model to the `HodModelFactory` via 
the ``baseline_model_instance`` keyword, the composite model instance is able to inform 
the factory of how to build a new instance of itself. 

Note that the use of the ``baseline_model_instance`` feature illustrated in 
:ref:`baseline_model_instance_mechanism_hod_building` can be applied 
equally well to additional features not present in the baseline model, 
and/or to replace features that already exist in the baseline model. 
For example, if you wanted to build an alternate version of ``zheng07`` in which 
the profiles of satellites were something different from NFW, then when passing in 
the ``baseline_model_instance`` you would also pass a ``satellites_profile`` keyword 
that contained the alternative profile model you want.  

This tutorial continues with :ref:`hod_modeling_tutorial3`. 







