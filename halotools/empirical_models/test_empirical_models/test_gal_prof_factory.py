#!/usr/bin/env python
from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc
from .. import gal_prof_factory as gpf
from astropy import cosmology

def test_unbiased_trivial():
	trivial_prof = hpc.TrivialProfile()
	gal_type = 'cens'

	cen_prof = gpf.GalProfModel(gal_type, trivial_prof)
	assert cen_prof.gal_type == gal_type

	assert isinstance(cen_prof.halo_prof_model, hpc.TrivialProfile)

	assert isinstance(cen_prof.cosmology, cosmology.FlatLambdaCDM)

	assert 0 <= cen_prof.redshift <= 100

	assert cen_prof.haloprop_key_dict == {}

	assert hasattr(cen_prof,'spatial_bias_model')

	assert cen_prof.param_dict == {}

	assert cen_prof.gal_prof_func_dict == {}

