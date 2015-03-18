#!/usr/bin/env python
from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc

__all__ = ['test_NFWProfile_SpatialBias']

def test_NFWProfile_SpatialBias():
	""" Function testing the implementation of a spatially biased NFW profile. 

	Specifically, `test_NFWProfile_SpatialBias` tests how 
	`~halotools.empirical_models.gal_prof_components.SpatialBias` 
	performs when operating on 
	`~halotools.empirical_models.halo_prof_components.NFWProfile`

	"""
	nfw = hpc.NFWProfile()
	gal_type = 'sats'
	biased_prof_param_list = nfw.prof_param_keys

	biased_nfw = gpc.SpatialBias(gal_type, nfw, 
		input_prof_params=biased_prof_param_list)

	assert type(biased_nfw.multiplicative_bias)==bool