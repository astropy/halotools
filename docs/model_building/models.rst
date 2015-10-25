
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

If you are looking for a tutorial on a specific component model, 
see :ref:`model_components_tutorials`. 

Reference/API
=============

.. automodapi:: halotools.empirical_models
.. automodapi:: halotools.empirical_models.test_empirical_models














