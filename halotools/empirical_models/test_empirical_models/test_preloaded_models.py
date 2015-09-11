#!/usr/bin/env python

from unittest import TestCase
import pytest
import numpy as np
from astropy.table import Table
from .. import preloaded_models
from ...utils.table_utils import compute_conditional_percentiles
from ...sim_manager import HaloCatalog

### Determine whether the machine is mine
# This will be used to select tests whose 
# returned values depend on the configuration 
# of my personal cache directory files
from astropy.config.paths import _find_home 
aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

class TestHearin15(TestCase):

	def setup_class(self):

		Npts = 1e4
		mass = np.zeros(Npts) + 1e12
		conc = np.random.random(Npts)
		d = {'halo_mvir': mass, 'halo_nfw_conc': conc}
		self.toy_halo_table = Table(d)
		self.toy_halo_table['halo_nfw_conc_percentile'] = compute_conditional_percentiles(
			halo_table = self.toy_halo_table, 
			prim_haloprop_key = 'halo_mvir', 
			sec_haloprop_key = 'halo_nfw_conc', 
			dlog10_prim_haloprop = 0.05)

		highz_mask = self.toy_halo_table['halo_nfw_conc_percentile'] >= 0.5
		self.highz_toy_halos = self.toy_halo_table[highz_mask]
		self.lowz_toy_halos = self.toy_halo_table[np.invert(highz_mask)]

		self.snapshot = HaloCatalog(preload_halo_table = True)

	@pytest.mark.skipif('not APH_MACHINE')
	def test_Hearin15(self):

		model = preloaded_models.Hearin15(concentration_binning = (1, 35, 5))
		model.populate_mock(snapshot = self.snapshot)

	def test_Leauthaud11(self):

		model = preloaded_models.Leauthaud11(concentration_binning = (1, 35, 5))
		model.populate_mock(snapshot = self.snapshot)

		model2 = preloaded_models.Leauthaud11(concentration_binning = (1, 35, 5), 
			central_velocity_bias = True, satellite_velocity_bias = True)
		model2.param_dict['velbias_centrals'] = 10
		model2.populate_mock(snapshot = self.snapshot)

		# Test that the velocity bias is actually operative
		central_mask = ( 
			(model.mock.galaxy_table['gal_type'] == 'centrals') & 
			(model.mock.galaxy_table['halo_mvir'] > 5e12) & 
			(model.mock.galaxy_table['halo_mvir'] > 1e13)
			)
		cens1 = model.mock.galaxy_table[central_mask]

		central_mask = ( 
			(model2.mock.galaxy_table['gal_type'] == 'centrals') & 
			(model2.mock.galaxy_table['halo_mvir'] > 5e12) & 
			(model2.mock.galaxy_table['halo_mvir'] > 1e13)
			)
		cens2 = model2.mock.galaxy_table[central_mask]

		assert np.std(cens1['vx']) < np.std(cens2['vx'])
		assert np.std(cens1['vy']) < np.std(cens2['vy'])
		assert np.std(cens1['vz']) < np.std(cens2['vz'])






