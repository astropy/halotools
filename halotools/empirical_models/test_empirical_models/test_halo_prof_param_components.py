#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
__all__ = ['test_ConcMass']

import numpy as np

from .. import halo_prof_param_components
from astropy import cosmology
from astropy.table import Table

from ...sim_manager import FakeSim, FakeMock

def test_ConcMass():
	""" Test the `~halotools.empirical_models.halo_prof_param_components.ConcMass` module. 
	Summary of tests is as follows: 
	
		* Returned concentrations satisfy :math:`0 < c < 100` for the full range of reasonable masses

		* Returns identical results regardless of argument choice

		* The :math:`c(M)` relation is monotonic over the full range of reasonable masses

	"""
	default_model = halo_prof_param_components.ConcMass()
	assert hasattr(default_model, 'cosmology')
	assert isinstance(default_model.cosmology, cosmology.FlatLambdaCDM)
	assert hasattr(default_model, 'redshift')
	assert hasattr(default_model, 'prim_haloprop_key')

	Npts = 1e3
	mass = np.logspace(10, 15, Npts)
	conc = default_model(prim_haloprop=mass)
	assert np.all(conc > 1)
	assert np.all(conc < 100)
	assert np.all(np.diff(conc) < 0)

	fake_sim = FakeSim()
	fake_mock = FakeMock()
	model_z0 = halo_prof_param_components.ConcMass(prim_haloprop_key = 'mvir', redshift=0)
	conc_z0_arg1 = model_z0(prim_haloprop = fake_sim.halos[model_z0.prim_haloprop_key])
	conc_z0_arg2 = model_z0(halos = fake_sim.halos)
	assert np.all(conc_z0_arg1 == conc_z0_arg2)
	conc_z0_arg3 = model_z0(galaxy_table = fake_mock.galaxy_table)
	assert np.all(conc_z0_arg3 == model_z0(prim_haloprop=fake_mock.galaxy_table['halo_mvir']))








