.. _composing_new_models:

**********************************************************************
Tutorial on designing your own model of the galaxy-halo connection
**********************************************************************

By following this tutorial, you will learn how to use the 
Halotools framework to design your own model of the galaxy-halo connection. 

The `~halotools.empirical_models` factories come in two different types, 
HOD-style models and subhalo-based models. In HOD-style models, 
there is no connection between the abundance of satellite 
galaxies in a halo and the number of subhalos. In these models, 
satellite abundance in each halo is determined by a Monte Carlo realization 
of some analytical model. In subhalo-based models, there is a one-to-one 
correspondence between subhalos and satellite galaxies. 

The implementation of these two different kinds of models is sufficiently 
different that the modeling for each is done by a separate factory. 
The `~halotools.empirical_models.HodModelFactory` builds HOD-style models, 
and the `~halotools.empirical_models.SubhaloModelFactory` builds 
subhalo-based models. Choose which class of model you are most interested in 
and follow the link to the appropriate tutorial:

.. toctree::
   :maxdepth: 1

   abunmatch_model_factory_overview
   hod_model_factory_overview
   

