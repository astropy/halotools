:orphan:

.. currentmodule:: halotools.empirical_models

.. _building_your_own_galaxy_halo_model:

***************************************************************
Building your own Model of the Galaxy-Halo Connection
***************************************************************

This section of the documentation gives an overview of how 
the galaxy-halo connection is modeled in Halotools. 


Basic Terminology: Composite Models and Component Models 
==========================================================

Let us begin by defining some Halotools-specific terminology:

* A **composite model** is a complete description of the mapping(s) between dark matter halos and *all* properties of their resident galaxy population. A composite model provides sufficient information to populate an ensemble of halos with a Monte Carlo realization of a galaxy population. Such a population constitutes the fundamental observable prediction of the model.  

* A **component model** provides a map between dark matter halos and a single property of the resident galaxy population. Examples include the stellar-to-halo mass relation, an NFW radial profile and the halo mass-dependence of the quenched fraction. 

Model Factory Design Overview
==========================================================

Halotools composite models are composed of a collection of independently defined component models. The composition is handled with an object-oriented factory design pattern. The basic way this works is as follows. 

1. Choose the set of features of the galaxy population you want to model. This could include luminosity, star formation rate, bulge-to-disk ratio, orientation angle, etc. There are no limits on the number of component models you can use. 

2. For each feature, either choose a Halotools-provided class, build your own version of the feature by writing your own subclass of the appropriate Halotools template, or write your own class from scratch. You are free to choose any permutation of these options you like. 

3. Once you have instances of each of these classes, pass the collection of instances to the appropriate Halotools model factory. The model factories create a standardized interface connecting halo catalogs, galaxy  models and observational predictions. 

After you instantiate a Halotools model factory, the resulting object has all the information it needs to generate  that can be directly compared to observational measurements. The set of methods bound to each composite model will vary depending on the features you choose. For example, if one of your chosen component models has a *quenched_fraction_vs_halo_mass* method, then so too will your composite model. However, composite models also have a uniform syntax for making mock catalogs. So no matter what features for your galaxies that you choose to model, *all* composite models have the following method:

>>> from halotools.sim_manager import CachedHaloCatalog
>>> halocat = CachedHaloCatalog() # doctest: +SKIP
>>> component_model.populate_mock(halocat) # doctest: +SKIP

The ``populate_mock`` method can be used to generate a Monte Carlo realization of the galaxy distribution into *any* Halotools-formatted catalog, which includes the pre-processed catalogs provided by Halotools (see :ref:`working_with_halotools_provided_catalogs` for more information), or halo catalogs you provide yourself (see :ref:`working_with_alternative_catalogs` for more information). 

If you are building an HOD-style model, in which there is no connection between the abundance of satellite galaxies and the abundance of subhalos, then `~halotools.empirical_models.HodModelFactory` is in charge of the component composition; see the :ref:`building_your_own_hod_model` section of the documentation for more information. 
If you are building an abundance matching-style model, in which there is one-to-one connection between galaxies and (sub)halos, then `~halotools.empirical_models.SubhaloModelFactory` is in charge of the component composition; see the :ref:`building_your_own_subhalo_model` section of the documentation for more information. 










