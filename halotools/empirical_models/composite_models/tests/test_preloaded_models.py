#!/usr/bin/env python

from unittest import TestCase
import pytest
import numpy as np
from astropy.table import Table

from ...composite_models import *
from ...factories import HodModelFactory, SubhaloModelFactory
from ...factories import PrebuiltHodModelFactory, PrebuiltSubhaloModelFactory

from ....utils.table_utils import compute_conditional_percentiles
from ....sim_manager import MarfMarfMarf
from ....custom_exceptions import *

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

	@pytest.mark.slow
	@pytest.mark.skipif('not APH_MACHINE')
	def setup_class(self):

		Npts = 1e4
		mass = np.zeros(Npts) + 1e12
		conc = np.random.random(Npts)
		d = {'halo_mvir': mass, 'halo_nfw_conc': conc}
		self.toy_halo_table = Table(d)
		self.toy_halo_table['halo_nfw_conc_percentile'] = compute_conditional_percentiles(
			table = self.toy_halo_table, 
			prim_haloprop_key = 'halo_mvir', 
			sec_haloprop_key = 'halo_nfw_conc', 
			dlog10_prim_haloprop = 0.05)

		highz_mask = self.toy_halo_table['halo_nfw_conc_percentile'] >= 0.5
		self.highz_toy_halos = self.toy_halo_table[highz_mask]
		self.lowz_toy_halos = self.toy_halo_table[np.invert(highz_mask)]

		self.halocat = MarfMarfMarf(preload_halo_table = True)

		self.halocat2 = MarfMarfMarf(preload_halo_table = True, redshift = 2.)

	@pytest.mark.slow
	@pytest.mark.skipif('not APH_MACHINE')
	def test_Hearin15(self):

		model = PrebuiltHodModelFactory('hearin15', concentration_binning = (1, 35, 5))
		model.populate_mock(halocat = self.halocat)

	@pytest.mark.slow
	@pytest.mark.skipif('not APH_MACHINE')
	def test_Leauthaud11(self):

		model = PrebuiltHodModelFactory('leauthaud11', concentration_binning = (1, 35, 5))
		model.populate_mock(halocat = self.halocat)

		# Test that an attempt to repopulate with a different halocat raises an exception
		with pytest.raises(HalotoolsError) as exc:
			model.populate_mock(redshift=2)
		with pytest.raises(HalotoolsError) as exc:
			model.populate_mock(simname='consuelo')
		with pytest.raises(HalotoolsError) as exc:
			model.populate_mock(halo_finder='bdm')

		model_highz = PrebuiltHodModelFactory('leauthaud11', redshift = 2., 
			concentration_binning = (1, 35, 5))
		model_highz.populate_mock(halocat = self.halocat2)
		with pytest.raises(HalotoolsError) as exc:
			model_highz.populate_mock()
		with pytest.raises(HalotoolsError) as exc:
			model_highz.populate_mock(halocat = self.halocat)
		model_highz.populate_mock(redshift = 2.)









