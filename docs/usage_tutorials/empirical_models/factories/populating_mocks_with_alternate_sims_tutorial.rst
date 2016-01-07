:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _populating_mocks_with_alternate_sims_tutorial:

****************************************************
Choosing the Simulation for your Mock Galaxy Catalog
****************************************************

Halotools factories give you the freedom to choose the simulation 
into which you will map galaxies onto dark matter halos. In particular, 
you can use 

	1. a pre-processed halo catalog downloaded by Halotools, 
	2. a "raw" halo catalog processed with the Halotools `~halotools.sim_manager` sub-package, or 
	3. your own catalog converted into a format recognized by Halotools using the `~halotools.sim_manager.MarfMarfMarf` class.

This section of the documentation covers how to make mocks using each of these three options. 

