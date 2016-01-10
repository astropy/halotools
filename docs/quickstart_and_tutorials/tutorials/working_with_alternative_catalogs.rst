:orphan:

.. _working_with_alternative_catalogs:

********************************************************
Working with Alternative Halo and Particle Catalogs
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
but do not to store the catalog in cache, you should instead use the stand-alone 
`~halotools.sim_manager.TabularAsciiReader` class. 


Using alternative catalogs 
===============================================================

Any halo-finder or simulation works perfectly well with Halotools. 
See :ref:`user_supplied_halo_catalogs`. 
