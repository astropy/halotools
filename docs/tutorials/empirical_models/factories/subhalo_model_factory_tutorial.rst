:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _subhalo_model_factory_tutorial:

****************************************************
Tutorial on the SubhaloModelFactory Class
****************************************************

This section of the documentation provides detailed notes 
on the source code implementation of the `SubhaloModelFactory` class. 
The purpose of the `SubhaloModelFactory` class is to provide a flexible, standardized platform for building subhalo-based models that can directly populate simulations with mock galaxies. The goal is to make it easy to swap new modeling features in and out of the framework while maintaining a uniform syntax. This way, when you want to study one particular feature of the galaxy-halo connection, you can focus exclusively on developing that feature, leaving the factory to take care of the remaining aspects of the mock population. This tutorial describes in detail how the `~SubhaloModelFactory` accomplishes this standardization. 


Outline 
========

We will start in :ref:`subhalo_model_factory_design_overview` with a high-level 
description of how the class creates a composite model from 
a set of independently-defined features. 


.. _subhalo_model_factory_design_overview:

Overview of the factory design
=================================

The `SubhaloModelFactory` has virtually no behavior of its own; 
it should instead be thought of as a container class that collects together 
behaviors that are defined elsewhere. These behaviors are defined in 
*component models*, which are instances of Halotools classes that typically provide 
a single, specialized mapping between halos and some specific galaxy property. 
By composing these individual mappings together, 
the output of the factory is a composite model for the galaxy-halo connection in which 
any number of user-defined galaxy properties is simultaneously modeled. 

Although there are numerous options 
for the form of the arguments passed to `SubhaloModelFactory`, 
the basic input is a *model dictionary*. 
A model dictionary is just an ordinary python dictionary that stores the collection of 
component model instances whose behaviors are being unified together by the factory. 
The model dictionary contains all of the necessary information inform the `SubhaloModelFactory` 
how to build a composite model from the components. 

Each component model in a model dictionary typically has each of the following three private attributes:

	1. `_methods_to_inherit`

	2. `_galprop_dtypes_to_allocate`

	3. `_mock_generation_calling_sequence`

Each of these three attributes will be explained in detail below. Briefly, the 
`_methods_to_inherit` is a list of strings that instructs the `~SubhaloModelFactory` 
which methods in the component model should be carried over into the composite model. 
The `_galprop_dtypes_to_allocate` attribute is used to instruct the `~SubhaloMockFactory` 
of the shape and name of every Numpy array that should be allocated for every galaxy property 
assigned by the component model. The `_mock_generation_calling_sequence` specifies the sequential 
order in which the methods of the component model should be called by the composite model 
during mock population. 

Again, we will discuss these and other bookkeeping devices in more detail below. 
For now, simply observe what is accomplished by these three pieces of information. 
Each component model is effectively giving the factory the following message:
"I want you to know about the following methods, and only the following methods, and I will take care of how they will be computed: `_methods_to_inherit`; 
I need you to make sure that the when you call these methods, the following arrays that will be passed to them: `_galprop_dtypes_to_allocate`; when you use me to make a mock, I need you to call these
methods in the following sequence: `_mock_generation_calling_sequence`". In this way, not only is all the physically relevant behavior defined in the component models, but the component models themselves provide the instructions for how they should be used. The task of the `~SubhaloModelFactory` is simply to follow these instructions, and to ensure that mutually consistent messages are received from the set of components in the model dictionary. 












