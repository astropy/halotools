#!/usr/bin/env python
from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc

def test_nfw_spatial_bias_instance():
	nfw = hpc.NFWProfile()
	gal_type = 'sats'
	biased_prof_param_list = nfw.prof_param_keys

	biased_nfw = gpc.SpatialBias(gal_type, nfw, 
		input_prof_params=biased_prof_param_list)

	assert type(biased_nfw.multiplicative_bias)==bool