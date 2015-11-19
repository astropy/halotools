:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _subhalo_model_factory_tutorial:

****************************************************
Tutorial on the SubhaloModelFactory Class
****************************************************

This section of the documentation provides detailed notes 
on the source code implementation of the `SubhaloModelFactory` class. 
The purpose of the `SubhaloModelFactory` class is to provide a flexible, standardized platform for building subhalo-based models that can directly populate simulations with mock galaxies. The goal is to make it easy to swap new modeling features in and out of the framework while maintaining a uniform syntax. This way, when you want to study one particular feature of the galaxy-halo connection, you can focus exclusively on developing that feature, leaving the factory to take care of the remaining aspects of the mock population. This tutorial describes in detail how the `~SubhaloModelFactory` accomplishes that standardization. 


Outline 
========

We will start in :ref:`subhalo_model_factory_design_overview` with a high-level 
description of how the class creates a composite model from 
a set of independently-defined features. 


.. _subhalo_model_factory_design_overview:

Overview of the factory design
=================================
















