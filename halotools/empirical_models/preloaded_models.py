# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from . import model_factories
from . import preloaded_hod_blueprints

__all__ = ['Kravtsov04']

def Kravtsov04(**kwargs):
	""" Simple HOD-style model based on Kravtsov et al. (2004). 

	There are two populations, centrals and satellites. 
	Central occupation statistics are given by a nearest integer distribution 
	with first moment given by an ``erf`` function. 
	Satellite occupation statistics are given by a Poisson distribution 
	with first moment given by a power law that has been truncated at the low-mass end. 

	Under the hood, this model is built from a set of component models whose 
	behavior is coded up elsewhere. The behavior of the central occupations 
	derives from the `~halotools.empirical_models.hod_components.Zheng07Cens` class, while for 
	satellites the relevant class is `~halotools.empirical_models.hod_components.Kravtsov04Sats`. 

	This composite model was built by the `~halotools.empirical_models.model_factories.HodModelFactory`, 
	which followed the instructions contained in 
	`~halotools.empirical_models.Kravtsov04_blueprint`. 

	Parameters 
	----------
	threshold : float, optional keyword argument
		Luminosity threshold of the galaxy sample being modeled. 

	Returns 
	-------
	model : object 
		Instance of `~halotools.empirical_models.model_factories.HodModelFactory`

	Examples 
	--------
	>>> model = Kravtsov04()
	>>> model = Kravtsov04(threshold = -20.5)

	"""
	blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint(**kwargs)
	return model_factories.HodModelFactory(blueprint, **kwargs)
















