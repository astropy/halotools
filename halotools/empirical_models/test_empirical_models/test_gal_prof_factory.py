#!/usr/bin/env python
import numpy as np 

from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc
from .. import gal_prof_factory as gpf
from ..mock_factory import HodMockFactory

from ...sim_manager.generate_random_sim import FakeSim
from ..preloaded_models import Kravtsov04

from astropy import cosmology


__all__ = ['test_unbiased_trivial', 'test_unbiased_nfw']

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

def test_unbiased_nfw():
	nfw_prof = hpc.NFWProfile()
	gal_type = 'satellites'

	sat_prof = gpf.GalProfFactory(gal_type, nfw_prof)

	snapshot = FakeSim()
	composite_model = Kravtsov04()
	mock = HodMockFactory(snapshot, composite_model)

	# Check that mc_radii gives reasonable results for FakeSim
	satellite_boolean = mock.gal_type == gal_type
	conc_key = mock.model.gal_prof_param_list[0]
	satellite_conc = getattr(mock, conc_key)[satellite_boolean]
	satellite_radii = sat_prof.mc_radii(satellite_conc)
	assert np.all(satellite_radii < 1)
	assert np.all(satellite_radii > 0)
	# Check that mc_radii scales properly for high- and low-concentrations
	conc_array_ones = np.ones_like(satellite_conc)
	conc_array_tens = np.ones_like(satellite_conc)*10
	high_conc_radii = sat_prof.mc_radii(conc_array_tens)
	low_conc_radii = sat_prof.mc_radii(conc_array_ones)
	assert high_conc_radii.mean() < satellite_radii.mean() < low_conc_radii.mean()

	# verify that all mc_angles points are on the unit sphere
	unit_sphere_pts = sat_prof.mc_angles(1000)
	norms = np.linalg.norm(unit_sphere_pts, axis=1)
	assert np.allclose(norms, 1)

	# verify that all mc_pos points are inside the unit sphere
	satellite_pos = sat_prof.mc_pos(mock)
	assert np.all(np.linalg.norm(satellite_pos, axis=1) <= 1)














