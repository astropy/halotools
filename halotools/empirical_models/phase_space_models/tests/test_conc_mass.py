#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
__all__ = ['test_ConcMass']

import numpy as np
from astropy import cosmology
from astropy.table import Table

from ..conc_mass_models import ConcMass

from ....sim_manager import FakeSim, FakeMock

def test_ConcMass():
	""" Test the `~halotools.empirical_models.ConcMass` module. 
	Summary of tests is as follows: 
	
		* Returned concentrations satisfy :math:`0 < c < 100` for the full range of reasonable masses

		* Returns identical results regardless of argument choice

		* The :math:`c(M)` relation is monotonic over the full range of reasonable masses

	"""
	default_model = ConcMass(redshift=0)

	Npts = 1e3
	mass = np.logspace(10, 15, Npts)
	conc = default_model.compute_concentration(prim_haloprop=mass)
	assert np.all(conc > 1)
	assert np.all(conc < 100)
	assert np.all(np.diff(conc) < 0)

	# fake_sim = FakeSim()
	# fake_mock = FakeMock()
	# model_z0 = ConcMass(prim_haloprop_key = 'halo_mvir', redshift=0)
	# conc_z0_arg1 = model_z0.compute_concentration(prim_haloprop = fake_sim.halo_table[model_z0.prim_haloprop_key])
	# conc_z0_arg2 = model_z0.compute_concentration(halo_table = fake_sim.halo_table)
	# assert np.all(conc_z0_arg1 == conc_z0_arg2)
	# conc_z0_arg3 = model_z0.compute_concentration(halo_table = fake_mock.galaxy_table)
	# assert np.all(conc_z0_arg3 == model_z0.compute_concentration(prim_haloprop=fake_mock.galaxy_table['halo_mvir']))








