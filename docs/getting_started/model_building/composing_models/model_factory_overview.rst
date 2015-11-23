:orphan:

.. _model_factory_overview:

****************************************************************
Overview of the Model-Building Factories
****************************************************************

In this section of the documentation, we will give a detailed description of the design pattern underlying Halotools models. The material covered here will be sufficient for you to understand how to use the Halotools framework to build your own customized model of the galaxy-halo connection. 

Halotools employs a factory design pattern to build *composite models* from a set of independently-defined *component models*. A component model governs one particular feature of one particular galaxy population. For example, 
`~halotools.empirical_models.NFWPhaseSpace` class is a component model that governs the intra-halo positions and velocities of satellites; `~halotools.empirical_models.Behroozi10SmHm` governs the stellar mass of centrals. On the other hand, a composite model governs *all* of the features of *every* galaxy population in your model universe. 

.. _model_factory_flowchart:

Flowchart of the model factory design pattern
-----------------------------------------------

The cartoon diagram below gives a visual sketch of the factory pattern by which composite models are built from a set of components. 

.. image:: model_factory_flowchart.png

The right side of the diagram is an information flowchart; the left side tells you the type of object used in each step. To build a model model, what you actually build is a composite model dictionary (a python dictionary) and then pass that dictionary to the appropriate model factory class (a python class). Your dictionary is just passed as an input to the python class constructor in the usual way: as the argument to `__init__`. The factory class then interprets your dictionary as a set of instructions for how to build an instance of that model. Once you instantiate the model factory class, the resulting instance has all of the tools you need to generate a synthetic galaxy catalogs based on your model.

Once you know how to build a dictionary for your model, passing that dictionary to the model factory is as simple as the usual process of instantiating a python class. 
For documentation on building dictionarys for HOD/CLF-style models, see :ref:`hod_model_factory_overview`; for documentation on building abundance matching-style models (i.e., any model that uses subhalos), see :ref:`abunmatch_model_factory_overview`.

