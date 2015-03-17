#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
__all__ = ['test_ConcMass']

import numpy as np

from .. import halo_prof_param_components
from astropy import cosmology

def test_ConcMass():
	default_model = halo_prof_param_components.ConcMass()
	assert isinstance(default_model.cosmology, cosmology.FlatLambdaCDM)

	Npts = 1e3
	mass = np.logspace(10, 15, Npts)
	conc = default_model.conc_mass(mass)
	assert np.all(conc > 1)
	assert np.all(conc < 100)
	assert np.all(np.diff(conc) < 0)
