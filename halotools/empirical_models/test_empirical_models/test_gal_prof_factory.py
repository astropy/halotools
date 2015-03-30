#!/usr/bin/env python
import numpy as np 

from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc
from .. import gal_prof_factory as gpf
from ..mock_factory import HodMockFactory

from ...sim_manager.generate_random_sim import FakeSim
from ..preloaded_models import Kravtsov04

from astropy import cosmology

def test_unbiased_trivial():
	trivial_prof = hpc.TrivialProfile()
	gal_type = 'centrals'

	cen_prof = gpf.GalProfFactory(gal_type, trivial_prof)
	assert cen_prof.gal_type == gal_type

	assert isinstance(cen_prof.halo_prof_model, hpc.TrivialProfile)

	assert isinstance(cen_prof.cosmology, cosmology.FlatLambdaCDM)

	assert 0 <= cen_prof.redshift <= 100

	assert cen_prof.haloprop_key_dict == {}

	assert hasattr(cen_prof,'spatial_bias_model')

	assert cen_prof.param_dict == {}

	assert cen_prof.gal_prof_func_dict == {}

	snapshot = FakeSim()
	composite_model = Kravtsov04()
	mock = HodMockFactory(snapshot, composite_model)

	trivial_result = cen_prof.mc_pos(mock)
	assert np.all(trivial_result == 0)


