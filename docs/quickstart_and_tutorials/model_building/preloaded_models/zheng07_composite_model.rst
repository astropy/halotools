.. _zheng07_composite_model:

*********************************************
Zheng et al. (2007) Composite Model
*********************************************

Building the Model 
=====================
This HOD-style is based on Zheng et al. (2007), arXiv:0703457. 
You can build an instance of this model using the 
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model_instance = PrebuiltHodModelFactory('zheng07')

As with any Halotools composite model, the above line of code 
will return a model instance that can populate N-body simulations 
with mock galaxy catalogs using the following syntax:

>>> model_instance.populate_mock(simname = 'any_halotools_formatted_catalog') # doctest: +SKIP

Overview of the Model Features
=================================

There are two populations, centrals and satellites. 
Central occupation statistics are given by a nearest integer distribution 
with first moment given by an ``erf`` function; the class governing this 
behavior is `~halotools.empirical_models.occupation_components.Zheng07Cens`. 
Central galaxies are assumed to reside at the exact center of the host halo; 
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`. 

Satellite occupation statistics are given by a Poisson distribution 
with first moment given by a power law that has been truncated at the low-mass end; 
the class governing this behavior is `~halotools.empirical_models.occupation_components.Zheng07Sats`; 
satellites in this model follow an (unbiased) NFW profile, as governed by the 
`~halotools.empirical_models.NFWPhaseSpace` class. 


Customizing the Model
=================================

There are two keyword arguments you can use to customize 
the instance returned by the factory. 

First, the ``threshold`` keyword argument pertains to the r-band absolute magnitude 
of the luminosity of the galaxy sample. 










