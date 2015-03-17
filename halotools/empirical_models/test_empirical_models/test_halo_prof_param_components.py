#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
__all__ = ['test_ConcMass']

import numpy as np

from .. import halo_prof_param_components
from astropy import cosmology

def test_ConcMass():
	""" Test the `~halotools.empirical_models.halo_prof_param_components.ConcMass` module. 
	Summary of tests is as follows: 
	
		* Returned concentrations satisfy :math:`0 < c < 100` for the full range of reasonable masses

		* The :math:`c(M)` relation is monotonic over the full range of reasonable masses

	"""
	default_model = halo_prof_param_components.ConcMass()
	assert isinstance(default_model.cosmology, cosmology.FlatLambdaCDM)

	Npts = 1e3
	mass = np.logspace(10, 15, Npts)
	conc = default_model.conc_mass(mass)
	assert np.all(conc > 1)
	assert np.all(conc < 100)
	assert np.all(np.diff(conc) < 0)
