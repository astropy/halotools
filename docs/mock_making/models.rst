
.. _model_building:

*********************************************
Building models of the Galaxy-Halo connection
*********************************************

Halotools provides a broad range of options for 
studying the connection between galaxies and 
their dark matter halos. The package comes pre-loaded 
with a small collection of specific models with default 
settings tuned to provide reasonably realistic mock 
universes. There are model building modules allowing 
you create a composite model by 
composing the behavior of a set of component models. 
And it's straightforward to build one model and swap out 
individual features to create a companion model. We describe 
each of these three modes of model building below. 

Pre-loaded halo occupation models 
=================================
There are numerous specific models that come pre-built 
into the model building package. After importing 
the module, each pre-built model can be loaded into 
memory with a single line of code. 

	>>> from halotools import pre_loaded_models
	>>> kravtsov04 = pre_loaded_models.Kravtsov04()

This simple call with no arguments builds an 
instance of a model based on the formulation of the HOD introduced in 
Kravtsov, et al. 2004. There are several optional keyword arguments 
that allow you to toggle between different galaxy samples built by 
the Kravtsov04 class. For example, 

	>>> kravtsov04 = pre_loaded_models.Kravtsov04(luminosity_threshold=-18, redshift_space=True, colors='sdss')

For a complete listing of the optional features supported 
by this pre-built model, see the Kravtsov04 documentation.




Composing your own galaxy-halo model
====================================

Permuting the different component behaviors 
creates flexibilty to ask targeted questions about 
specific features in the galaxy distribution. 
The way this works in Halotools is most easily explained by example. 

	>>> from halotools.models import hod_designer

The HOD Designer class provides a blueprint 
that gives instructions to the model factories and mock 
factories. 



