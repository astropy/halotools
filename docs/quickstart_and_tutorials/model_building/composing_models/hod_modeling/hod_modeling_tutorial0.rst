
.. _hod_modeling_tutorial0:

****************************************************************
Tutorial on building an HOD-style model
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes how you can use the 
`HodModelFactory` class to build a custom-designed HOD-style model 
of the galaxy-halo connection. 
In the :ref:`preliminary_discussion_hod_factory_inputs` section below, 
you will learn about the basic calling sequence for the `HodModelFactory`; 
the material covered there applies to all HOD-style models. 
The remainder of the tutorial provides a sequence in increasingly complex 
examples of models you can build with the `HodModelFactory`. 
If you already have some familiarity with Halotools modeling, 
you may wish to skip ahead to the :ref:`hod_modeling_worked_examples_list` section. 


.. _preliminary_discussion_hod_factory_inputs:

Basic instructions for calling the `HodModelFactory`
=========================================================


The data structure of any composite model dictionary is a python dictionary. The set of key-value pairs in this dictionary constitutes the complete set of instructions required to build the composite model.  For HOD-style models, the keys of the composite model dictionary are the names of the galaxy population being modeled, e.g., `centrals` and `satellites`. The value of each key of an HOD-style composite model dictionary is itself a python dictionary; in Halotools lingo this second dictionary is called a *subpopulation dictionary*. This means that in HOD-style models, a composite model dictionary is actually a dictionary of dictionaries. So to build any HOD-style model, what you must do is build a collection of subpopulation dictionarys, and bundle the set of subpopulation dictionarys together into a composite model dictionary. The cartoon diagram below gives a schematic for the `~halotools.empirical_models.Zheng07` composite HOD model that we will look at in detail below. 


Let's start by looking at how the two subpopulation dictionarys are built, starting with the `satellites`. Like all python dictionaries, the satellite dictionary is specified by a set of key-value pairs. Each key of a subpopulation dictionary are just nicknames for the type feature; for the dictionary shown above, the Halotools convention is to use `occupation` for the feature governing the abundance of the galaxy type, while `profile` refers to the feature governing how the galaxy type is distributed within its dark matter halo. 

The value bound to each of the above keys are instances of Halotools classes, diagrammed here at the base of each arrow flowing into subpopulation dictionary. These class instances govern that particular feature. For example, suppose we wished for the spatial positions of satellites within their halos to follow an NFW profile. Then the value bound to the `profile` key of the satellite dictionary would be an instance of the `~halotools.empirical_models.NFWPhaseSpace` class. This particular feature is a class that is already in Halotools, but the framework described here provides you with the freedom to supplement Halotools with any feature(s) you write with your own code. 

.. _hod_modeling_worked_examples_list:

Worked examples of building HOD-style models
=========================================================

.. toctree::
   :maxdepth: 1

   hod_modeling_tutorial1 

