
.. _model_building:

*************************************************
Tutorial on modeling the galaxy-halo connection
*************************************************

This section of the documentation is the starting point for in-depth tutorials 
on the model-building factories, which give you two basic options:  

	1. **Use a pre-built model.** Each of the Halotools pre-built models is an implementation of some specific publication, has many options for customizing its behavior, and right out-of-the-box can populate mock catalogs and make a wide range of observational predictions. 

	2. **Design your own model.** You have the option to choose between Halotools-provided features, write all of your own features, or anywhere in between. 

Before diving in to the rest of the tutorial, it will be helpful to familiarize yourself with some Halotools-specific terminology.  

Basic Terminology: *composite models* and *component models*
==============================================================

* A **composite model** is a complete description of the mapping(s) between dark matter halos and *all* properties of their resident galaxy population. A composite model provides sufficient information to populate an ensemble of halos with a Monte Carlo realization of a galaxy population. Such a population constitutes the fundamental observable prediction of the model.  

* A **component model** provides a map between dark matter halos and a single property of the resident galaxy population. Examples include the stellar-to-halo mass relation, an NFW radial profile and the halo mass-dependence of the quenched fraction. 

With this terminology in mind, you should choose between the following two branches of this tutorial:

.. toctree::
   :maxdepth: 1

   preloaded_models/index
   composing_models/index










