# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from . import hod_factory
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
	derives from the `~halotools.empirical_models.hod_components.Kravtsov04Cens` class, while for 
	satellites the relevant class is `~halotools.empirical_models.hod_components.Kravtsov04Sats`. 

	This composite model was built by the `~halotools.empirical_models.hod_factory.HodModelFactory`, 
	which followed the instructions contained in 
	`~halotools.empirical_models.Kravtsov04_blueprint`. 

	"""
	blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint(**kwargs)
	return hod_factory.HodModelFactory(blueprint)
















