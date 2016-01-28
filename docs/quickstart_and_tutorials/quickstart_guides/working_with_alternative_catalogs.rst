:orphan:

.. _working_with_alternative_catalogs:

********************************************************
Working with alternative halo and particle catalogs
********************************************************

This section of the documentation describes how you can 
use Halotools with simulations besides the pre-processed snapshots 
that come standard with the package. 


Reducing a new Rockstar catalog 
===============================================================

The `~halotools.sim_manager.RockstarHlistReader` class allows you to 
read the ASCII output of the Rockstar halo-finder, apply row- and column-wise 
cuts of your choosing, and store the resulting catalog in the Halotools cache 
as a fast-loading hf5 file. If this is your use-case, see the 
:ref:`reducing_and_caching_a_new_rockstar_catalog` section of the documentation. 

If you want to reduce your own Rockstar catalog with Halotools 
but do not want to store the catalog in cache, you should instead use the stand-alone 
`~halotools.sim_manager.TabularAsciiReader` class. 


Using alternative catalogs 
===============================================================

The full functionality of Halotools is available for use with 
halos in any N-body simulation identified with any halo-finder, 
and use of the Halotools caching system is optional in every respect. 
For more information about how to 
work with simulation data not provided by Halotools, 
see :ref:`user_supplied_halo_catalogs`. 


Using your own collection of dark matter particles 
===============================================================

Some features in Halotools requires use of a catalog of 
a random sample of dark matter particles from the same snapshot as the 
halo catalog. Most of the Halotools-provided halo catalogs are accompanied by 
a corresponding particle catalog, but you may prefer to provide your own, 
or you may wish to use these features with alternative simulation data. 
See the :ref:`working_with_alternative_particle_data` section of the 
documentation for how to put your collection of particles into a standard form 
and optionally store the particle data in your cache. 













