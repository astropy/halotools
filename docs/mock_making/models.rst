
.. _model_building:

*********************************************
Building models of the Galaxy-Halo connection
*********************************************

.. currentmodule:: halotools.empirical_models

Halotools provides a broad range of options for 
studying the connection between galaxies and 
their dark matter halos. These options fall into 
one of three categories. First, the package comes pre-loaded 
with a small collection of specific models 
very similar to those in the published literature, 
including default values to the best-fit published values. 
Second, once you have any model in hand, 
it's straightforward to toggle the model parameters, and/or swap out 
individual features to create companion models. 
Finally, for the most flexibility, 
there are modules allowing you create a composite model by 
composing the behavior of a set of component models. 
We describe each of these three modes of model building below. 

Pre-loaded halo occupation models 
=================================
There are numerous specific models that come pre-built 
into the model building package. After importing 
the module, each pre-built model can be loaded into 
memory with a single line of code. 

	>>> from halotools.empirical_models import preloaded_models
	>>> kravtsov04 = preloaded_models.Kravtsov04()

This simple call with no arguments builds an 
instance of a model based on the formulation of the HOD introduced in 
Kravtsov, et al. (2004), with default settings to use 
best-fit parameter values taken from the subsequent literature. 
For a complete listing of the optional features supported by this pre-built model, 
see the `~halotools.empirical_models.Kravtsov04_blueprint` 
documentation `~halotools.empirical_models`.

.. _list_of_default_models: 

List of pre-loaded models 
--------------------------------

* Behroozi10, parameterized abundance matching
* Leauthaud11, an HOD model deriving from a central galaxy stellar-to-halo mass relation
* Zehavi11, red/blue luminosity-based HOD
* Kravtsov13, direct abundance matching
* Behroozi13, assembly history model based on abundance matching 
* Tinker13, similar to Leauthaud11, but with quenched and star-forming designations 

Composing your own galaxy-halo model
====================================

Permuting the different component behaviors 
creates flexibilty to ask targeted questions about 
specific features in the galaxy distribution. 
The way this works in Halotools is most easily explained by example. 

The HOD Designer class provides a blueprint for building a 
galaxy-halo model. After you write that blueprint, as described below, 
you pass it to a model factory, and you get back a 
you get a composite model. That's model that governs how 
galaxies are connected to halos. That galaxy-halo model object 
you get back has a built-in method to populate a mock universe with galaxies 
in a way that is determined by the parameter values of the model. 


Reference/API
=============

.. automodapi:: halotools.empirical_models
.. automodapi:: halotools.empirical_models.test_empirical_models














