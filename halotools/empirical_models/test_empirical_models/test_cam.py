#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 

from .. abunmatch import ConditionalAbunMatch
from .. import model_defaults
from ...sim_manager import FakeMock

def test_cam():
	galprop_key = 'gr_color'
	prim_galprop_key = 'stellar_mass'
	sec_haloprop_key = 'halo_zhalf'

	fake_mock = FakeMock()
	sm_min = fake_mock.galaxy_table['stellar_mass'].min()
	sm_max = fake_mock.galaxy_table['stellar_mass'].max()
	sm_bins = np.logspace(np.log10(sm_min)-0.01, np.log10(sm_max)+0.01, 50)

	cam = ConditionalAbunMatch(
		galprop_key=galprop_key, 
		prim_galprop_key = prim_galprop_key, 
		sec_haloprop_key = sec_haloprop_key, 
		input_galaxy_table = fake_mock.galaxy_table, 
		prim_galprop_bins = sm_bins
		)

	fake_mock2 = FakeMock()
	result = cam.mc_gr_color(galaxy_table = fake_mock2.galaxy_table)















