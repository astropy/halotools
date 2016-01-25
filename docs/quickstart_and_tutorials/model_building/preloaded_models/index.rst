.. _preloaded_models_overview:

*********************************************
Overview of Preloaded Models
*********************************************

This section of the documentation reviews the prebuilt composite models 
provided by Halotools. All prebuilt models can directly populate a simulation 
with a mock catalog and make observational predictions that can be compared 
to measurements made on a real galaxy sample. You need only choose the 
prebuilt model and simulation snapshot that is appropriate for your science application. 
By following the links below, you can 
find a step-by-step tutorial for how to get going with 
any of the Halotools-provided prebuilt models. 

Basic Terminology: Composite Models and Component Models 
==========================================================

Before diving in to the list of prebuilt models, we'll first begin by defining some Halotools-specific terminology. Users already familiar with Halotools lingo can skip ahead to the next section. 

* A **composite model** is a complete description of the mapping(s) between dark matter halos and *all* properties of their resident galaxy population. A composite model provides sufficient information to populate an ensemble of halos with a Monte Carlo realization of a galaxy population. Such a population constitutes the fundamental observable prediction of the model.  

* A **component model** provides a map between dark matter halos and a single property of the resident galaxy population. Examples include the stellar-to-halo mass relation, an NFW radial profile and the halo mass-dependence of the quenched fraction. 



Traditional HOD models
=========================

HOD-style models make predictions for galaxy samples selected 
to be brighter or more massive than some threshold lower bound. 
It is also common in the literature to apply this class of model 
to galaxy samples that have a more complex selection function 
such as red sequence galaxies or quasars. 

.. toctree::
   :maxdepth: 1

   zheng07_composite_model
   leauthaud11_composite_model


Decorated HOD models
============================
Decorated HOD models relax the assumption that halo mass is the only 
relevant halo property governing galaxy occupation statistics. 
This class of model naturally inherits all of the flexibility 
provided by the traditional HOD, while at the same time providing 
further flexibility to study the co-evolution of galaxies and their halos. 

.. toctree::
   :maxdepth: 1

   hearin15_composite_model

HOD models with color/SFR
==========================
While traditional HOD models have historically been applied to color-selected 
samples, such models do not contain any parameters controlling the dependence 
of color/SFR on halo properties. The HOD models in this section *do* 
contain such parametric freedom, making these models much more useful to 
constrain the physics of star formation and quenching. 

.. toctree::
   :maxdepth: 1

   tinker13_composite_model


Stellar-to-halo mass models (SMHM)
====================================
An SMHM model presumes that there is a one-to-one correspondence between 
stellar mass and subhalo mass. The SMHM relation is given by some parameterized form 
typically motivated by results based on abundance matching. 
As SMHM models are based on subhalos, they relax many of the "spherical cow" assumptions 
typical of HOD-style models. 

.. toctree::
   :maxdepth: 1

   behroozi10_composite_model


Abundance matching models 
===========================

.. toctree::
   :maxdepth: 1

   abundance_matching_composite_model




