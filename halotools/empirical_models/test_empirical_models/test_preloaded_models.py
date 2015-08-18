#!/usr/bin/env python

from unittest import TestCase
import numpy as np
from astropy.table import Table
from .. import preloaded_models
from ...utils.table_utils import compute_conditional_percentiles

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

	def test_default_model(self):

		model = preloaded_models.Hearin15()
