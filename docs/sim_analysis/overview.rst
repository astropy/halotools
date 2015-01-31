
.. _sim_analysis:

****************************************
Overview of simulation analysis tools 
****************************************

We now describe the functionality of Halotools 
related to the study of pure halo-level 
trends, in the absence of any galaxy evolution model ansatz. 

Catalog management 
--------------------

Halotools provides bookeeping tools to keep track 
of a potentially large amount of simulated data. 

	>>> from  halotools import catalog_manager as catman
	>>> print(catalog_manager.available_simulations)
	>>> bolshoi_snap = catman.retrieve_halos('bolshoi', z=0)
	>>> bolshoi_snap.galaxies.show()
	>>> bolshoi_snaplistf = catman.retrieve_particles('bolshoi', z=[0.02,0.25,0.5,0.8])
	>>> bolshoi_particles_z0 = catman.retrieve_particles('bolshoi_z0')

Those lines show that it is possible hold in your hand a full-on 
single-snapshot catalog of dark matter halos with just one line of code. 

Simulation list
===============

Bolshoi
--------

Here I colloquially describe Bolhoi. 