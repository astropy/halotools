#!/usr/bin/env python
import numpy as np

from ..hod_components import Leauthaud11Cens, Leauthaud11Sats

from .. import model_defaults

from astropy.table import Table
from copy import copy

__all__ = ['test_Leauthaud11Cens', 'test_Leauthaud11Sats']

def test_Leauthaud11Cens():
	""" Function to test 
	`~halotools.empirical_models.Leauthaud11Cens`. 
	"""

	model = Leauthaud11Cens()
	ncen1 = model.mean_occupation(prim_haloprop = 1.e12)

	mcocc = model.mc_occupation(prim_haloprop = np.ones(1e4)*1e12, seed=43)
	assert 0.5590 < np.mean(mcocc) < 0.5592

	model.param_dict['scatter_model_param1'] *= 1.5
	ncen2 = model.mean_occupation(prim_haloprop = 1.e12)
	assert ncen2 < ncen1

	model.param_dict['m10'] *= 1.1
	ncen3 = model.mean_occupation(prim_haloprop = 1.e12)
	assert ncen3 < ncen2

	model.param_dict['m11'] *= 1.1
	ncen4 = model.mean_occupation(prim_haloprop = 1.e12)
	assert ncen4 == ncen3


	model2 = Leauthaud11Cens(threshold = 10.75)
	ncen5 = model2.mean_occupation(prim_haloprop = 1.e12)
	assert ncen5 < ncen1

def test_Leauthaud11Sats():
	""" Function to test 
	`~halotools.empirical_models.Leauthaud11Cens`. 
	"""

	model = Leauthaud11Sats()





