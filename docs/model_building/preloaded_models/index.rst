.. _preloaded_models_overview:

*********************************************
Overview of Preloaded Models
*********************************************

There are numerous specific models that come pre-built 
into the model building package. After importing 
the module, each pre-built model can be loaded into 
memory with a single line of code. For example: 

	>>> from halotools.empirical_models import Zheng07
	>>> zheng07 = Zheng07()

All pre-built models can be used to directly populate dark matter halos 
with mock galaxies using a single line of code:

    >>> zheng07.populate_mock() # doctest: +SKIP

In this section of the documentation, we'll give a brief sketch of each 
preloaded model in the package. For a full exposition on each model's 
functionality, follow the link to the tutorial.


Traditional HOD models
=========================
Tutorial coming soon!

Assembly-biased HOD models
============================
Tutorial coming soon!


HOD models with color/SFR
==========================
Tutorial coming soon!


Stellar-to-halo mass models 
=============================
Tutorial coming soon!


Abundance matching models 
===========================
Tutorial coming soon!




