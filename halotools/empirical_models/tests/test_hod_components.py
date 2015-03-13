#!/usr/bin/env python
import numpy as np

from .. import preloaded_hod_blueprints
from .. import model_defaults
from .. import hod_components
from .. import gal_prof_factory



def test_Kravtsov04Cens():
	default_model = hod_components.Kravtsov04Cens()
	assert isinstance(default_model, hod_components.OccupationComponent)

	assert default_model.gal_type == 'centrals'
	correct_haloprops = {'halo_boundary', 'prim_haloprop_key'}
	assert set(default_model.haloprop_key_dict.keys()) == correct_haloprops
	assert default_model.num_haloprops == 1
	assert default_model.occupation_bound == 1
	assert default_model.prim_func_dict.keys() == [None]
	assert hasattr(default_model, 'mc_occupation')
	assert hasattr(default_model, 'mean_occupation')







