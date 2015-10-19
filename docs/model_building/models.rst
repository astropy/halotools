
.. _model_building:

*************************************************
Overview of modeling the Galaxy-Halo connection
*************************************************

.. currentmodule:: halotools.empirical_models

Halotools provides a broad range of options for 
studying the connection between galaxies and 
their dark matter halos. These options fall into 
one of three categories. First, the package comes pre-loaded 
with a small collection of specific models 
very similar to those in the published literature, 
including default values set to the best-fit published values. 
Second, once you have any model in hand, 
it's straightforward to toggle the model parameters, and/or swap out 
individual features to create companion models. 
Finally, for the most flexibility, 
there are modules allowing you create a composite model by 
composing the behavior of a set of component models. 
We describe each of these three modes of model-building below. 

.. toctree::
   :maxdepth: 1

   preloaded_models/index
   composing_models/index
   models_from_scratch/index
   

Pre-loaded halo occupation models 
=================================
There are numerous specific models that come pre-built 
into the model building package. After importing 
the module, each pre-built model can be loaded into 
memory with a single line of code. 

	>>> from halotools.empirical_models import preloaded_models
	>>> zheng07 = preloaded_models.Zheng07()

This simple call with no arguments builds an 
instance of a model based on the formulation of the HOD introduced in 
Kravtsov, et al. (2004), with default settings to use 
best-fit parameter values taken from the subsequent literature. 
For a complete listing of the optional features supported by this pre-built model, 
see the `~halotools.empirical_models.Zheng07` 
documentation `~halotools.empirical_models`.

.. _list_of_default_models: 

List of pre-loaded models 
--------------------------------

* `~halotools.empirical_models.Moster13SmHm` - parameterized abundance matching
* `~halotools.empirical_models.Zheng07` - simple HOD-style model based on arXiv:0703457.
* `~halotools.empirical_models.Leauthaud11` - HOD-style model based on arXiv:1103.2077 whose behavior derives from a central-galaxy stellar-to-halo mass relation

Many more models currently reside in development branches where they are being tested before 
incorporating into the main repository. Such models can always be provided to eager users upon request. 

Composing your own galaxy-halo model
====================================

Instructions coming soon!

Reference/API
=============

.. automodapi:: halotools.empirical_models
.. automodapi:: halotools.empirical_models.test_empirical_models














