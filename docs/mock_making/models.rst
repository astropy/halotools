
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
into the model building package. 

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



