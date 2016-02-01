.. _composing_new_models:

**********************************************************************
Tutorial on designing your own model of the galaxy-halo connection
**********************************************************************

By following this tutorial, you will learn how to use the 
Halotools framework to design your own model of the galaxy-halo connection. 
The `~halotools.empirical_models` factories come in two different types, 
*HOD-style models* and *subhalo-based models.* In HOD-style models, 
there is no connection between the abundance of satellite 
galaxies in a host halo and the number of subhalos in that host halo. In these models, 
satellite abundance in each halo is determined by a Monte Carlo realization 
of some analytical model. Examples of this approach to the galaxy-halo connection 
include the HOD, CLF and CSMF, as well as extensions of these that include additional 
features such as color-dependence. 

By contrast, in subhalo-based models there is a one-to-one 
correspondence between subhalos and satellite galaxies. In these models, 
every host halo in the simulation is connected to a central galaxy, 
and every subhalo is connected a satellite. Examples include traditional abundance 
matching, age matching, and parameterized stellar-to-halo mass models 
such as Behroozi et al. (2010) and Moster et al. (2010). 

The implementation of these two different approaches to the galaxy-halo connection 
is sufficiently different that the modeling for each is done by a separate factory. 
The `~halotools.empirical_models.HodModelFactory` builds HOD-style models, 
and the `~halotools.empirical_models.SubhaloModelFactory` builds 
subhalo-based models. Choose which class of model you are most interested in 
and follow the link to the appropriate tutorial:

.. toctree::
   :maxdepth: 1

   subhalo_modeling/subhalo_modeling_tutorial0
   hod_modeling/hod_modeling_tutorial0


