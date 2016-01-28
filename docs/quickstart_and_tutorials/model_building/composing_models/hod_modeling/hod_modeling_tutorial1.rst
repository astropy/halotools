.. _hod_modeling_tutorial1:

****************************************************************
Example 1: Building a simple HOD-style model
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes how you can use the 
`HodModelFactory` to build one of the simplest and most widely 
used galaxy-halo models in the literature: the HOD based on 
Zheng et al. (2007), arXiv:0703457. By following this example, 
you will learn the basic calling sequence for building HOD models. 
We will build an exact replica of the ``zheng07`` pre-built model, 
so you will also learn how the `PrebuiltHodModelFactory` is really just 
"syntax candy" for the `HodModelFactory`. 

Preliminary comments
=====================================================

In the HOD, the galaxy population is decomposed into distinct types: 
*centrals* and *satellites*. Models for the properties of these populations 
are typically very different, and so Halotools requires you to use 
independently defined component models for these two sub-populations. 
As described below, this is accomplished by your choices for the 
keyword arguments you spply to the `HodModelFactory`. 

As sketched in :ref:`factory_design_diagram`, 
to build any composite model you pass a 
set of component model instances to the `HodModelFactory`. 
These instances are all passed in the form of keyword arguments. 
The keywords you choose must be formatted in a specific way as these strings 
contain information that the factory uses to interpret each feature. 
In particular, the strings you use for keywords will be parsed in a way 
that informs the `HodModelFactory` of the name of galaxy type 
(e.g., 'centrals' or 'satellites'), as well as the kind of feature 
supplied by the component model. 

For the sake of having something concrete to hold on to, 
let's now dive in and write down the source code for how 
to call the `HodModelFactory`. 

Source code for the ``zheng07`` model
=====================================================

.. code:: python

	from halotools.empirical_models import HodModelFactory

	from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
	cens_occ_model =  Zheng07Cens(threshold = -20.5)
	cens_prof_model = TrivialPhaseSpace()

	from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
	sats_occ_model =  Zheng07Sats(threshold = -20.5)
	sats_prof_model = NFWPhaseSpace()

	model_instance = HodModelFactory(
		centrals_occupation = cens_occ_model, 
		centrals_profile = cens_prof_model, 
		satellites_occupation = sats_occ_model, 
		satellites_profile = sats_prof_model)


















