
.. _mock_observation_quickstart:

******************************************************************
Quickstart Guide to Making Observations of a Mock Galaxy Catalog
******************************************************************


Galaxy Clustering 
------------------

One of the simplest and most powerful observational statistics that can be used to 
constrain the galaxy-halo connection is the two-point correlation function of galaxies, 
aka galaxy clustering. For any model, all mock populations come with a 
built-in method to calculate galaxy clustering: 

>>> radial_bins, clustering = hod_model.mock.compute_galaxy_clustering() # doctest: +SKIP

Galaxy-Galaxy Lensing 
------------------------------------

Galaxies act as gravitational lenses on sources of background light behind them. 
The strength of this lensing signal is directly related to the strength of the cross-correlation 
between the population of galaxies (lenses) and the cosmic distribution of dark matter. 
For any model, all mock populations come with a built-in method to calculate this cross-correlation: 

>>> radial_bins, clustering = hod_model.mock.compute_galaxy_matter_cross_clustering() # doctest: +SKIP

Group Identification 
------------------------------------

Galaxies congregate into groups. The ideal group-finding algorithm would 
group galaxies together according to their true, underlying dark matter halos. In reality, 
no group-finder is perfect, particularly because of line-of-sight interpolers due to 
redshift-space distortions. There are a variety of algorithms with different 
levels of purity and incompleteness; to compute group membership of a mock galaxy population 
using a common friends-of-friends algorithm:

>>> groupIDs = hod_model.mock.compute_fof_group_ids() # doctest: +SKIP
