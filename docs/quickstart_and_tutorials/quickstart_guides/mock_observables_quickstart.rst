
.. _mock_observation_quickstart:

******************************************************************
Quickstart guide to making observations on your mock
******************************************************************


Galaxy Clustering 
------------------

One of the simplest and most powerful observational statistics that can be used to 
constrain the galaxy-halo connection is the two-point correlation function of galaxies, 
aka galaxy clustering. For any model, all mock populations come with a 
built-in method to calculate galaxy clustering: 

>>> radial_bins, clustering = hod_model.mock.compute_galaxy_clustering() # doctest: +SKIP

See `~halotools.empirical_models.MockFactory.compute_galaxy_clustering` for further documentation 
and more example usages. 

Galaxy-Galaxy Lensing 
------------------------------------

Galaxies act as gravitational lenses on sources of background light behind them. 
The strength of this lensing signal is directly related to the strength of the cross-correlation 
between the population of galaxies (lenses) and the cosmic distribution of dark matter. 
For any model, all mock populations come with a built-in method to calculate this cross-correlation: 

>>> radial_bins, clustering = hod_model.mock.compute_galaxy_matter_cross_clustering() # doctest: +SKIP

See `~halotools.empirical_models.MockFactory.compute_galaxy_matter_cross_clustering` for further documentation 
and more example usages. 


Redshift-Space Distortions 
-----------------------------

Description coming soon!



Galaxy Group Identification 
------------------------------------

Description coming soon!





