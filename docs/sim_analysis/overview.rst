
.. _sim_analysis:

****************************************
Overview of simulation analysis tools 
****************************************

Halotools can be used to study the assembly history and environmental 
trends of dark matter halos. 

.. _cat_manage:

Catalog management 
==================

Halotools provides bookeeping tools to keep track 
of a potentially large amount of simulated data, 
and also your various reductions of that data. 

	>>> from  halotools import catalog_manager as catman
	>>> print(catalog_manager.available_simulations)
	>>> bolshoi_snap = catman.retrieve_halos('bolshoi', z=0)
	>>> bolshoi_snap.galaxies.show()
	>>> bolshoi_snaplistf = catman.retrieve_particles('bolshoi', z=[0.02,0.25,0.5,0.8])
	>>> bolshoi_particles_z0 = catman.retrieve_particles('bolshoi_z0')

Those lines show that you can retrieve processed snapshot data 
from one of the built-in simulations with just one line of code. 
See :ref:`simulation_list` for more information about what simulations come pre-packaged 
with Halotools. 

.. _lss_analysis:

Studying structure formation with N-body simulations
====================================================

Dark matter simulations of cosmological structure formation 
are interesting in their own right,
and halotools provides modules that you can use to study them. 
For example, suppose you are interested in the following question: 
how does the tidal environment of a dark matter halo 
impact its mass accretion rate? 

Reference/API
=============

.. automodapi:: halotools.sim_manager