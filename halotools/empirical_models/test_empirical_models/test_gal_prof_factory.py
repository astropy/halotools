#!/usr/bin/env python
import numpy as np 

from .. import halo_prof_components as hpc
from .. import gal_prof_components as gpc
from .. import gal_prof_factory as gpf
from ..mock_factories import HodMockFactory

from ...sim_manager.generate_random_sim import FakeSim
from ..preloaded_models import Kravtsov04

from astropy import cosmology


__all__ = ['test_unbiased_trivial', 'test_unbiased_nfw']

def test_unbiased_trivial():

	cen_prof = gpf.SphericallySymmetricGalProf(
		gal_type='centrals', halo_prof_model=hpc.TrivialProfile)
	assert cen_prof.gal_type == 'centrals'

	assert isinstance(cen_prof.halo_prof_model, hpc.TrivialProfile)

	if hasattr(cen_prof, 'cosmology'):
		assert isinstance(cen_prof.cosmology, cosmology.FlatLambdaCDM)

	if hasattr(cen_prof, 'redshift'):
		assert 0 <= cen_prof.redshift <= 100

	assert hasattr(cen_prof, 'halo_boundary')
	assert hasattr(cen_prof, 'prim_haloprop_key')

	assert cen_prof.param_dict == {}

	snapshot = FakeSim()
	composite_model = Kravtsov04()
	mock = HodMockFactory(snapshot=snapshot, model=composite_model)

	x, y, z = cen_prof.mc_pos(galaxy_table=mock.galaxy_table)
	assert np.all(x == 0)
	assert np.all(y == 0)
	assert np.all(z == 0)

def test_unbiased_nfw():

	sat_prof = gpf.SphericallySymmetricGalProf(
		halo_prof_model=hpc.NFWProfile, gal_type='satellites')

	snapshot = FakeSim()
	composite_model = Kravtsov04()
	mock = HodMockFactory(snapshot=snapshot, model=composite_model)

	# Check that mc_radii gives reasonable results for FakeSim
	satellite_boolean = mock.galaxy_table['gal_type'] == sat_prof.gal_type
	conc_key = 'NFWmodel_conc'
	satellite_conc = mock.galaxy_table[conc_key][satellite_boolean]
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
	x, y, z = sat_prof.mc_angles(1000)
	unit_sphere_pts = np.array([x, y, z]).T
	norms = np.linalg.norm(unit_sphere_pts, axis=1)
	assert np.allclose(norms, 1)

	# verify that all mc_pos points are inside the unit sphere
	satellite_xpos, satellite_ypos, satellite_zpos = sat_prof.mc_pos(galaxy_table=mock.galaxy_table)
	satellite_pos = np.array([satellite_xpos, satellite_ypos, satellite_zpos]).T
	assert np.all(np.linalg.norm(satellite_pos, axis=1) <= 1)














