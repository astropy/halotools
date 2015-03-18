#!/usr/bin/env python
from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc
from .. import gal_prof_factory as gpf

def test_gal_prof_factory_instance():
	nfw = hpc.NFWProfile()
	gal_type = 'sats'

	gal_prof_model = gpf.GalProfModel(gal_type, nfw)
	assert gal_prof_model.gal_type == gal_type	