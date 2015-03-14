#!/usr/bin/env python

import numpy as np 
from .. import preloaded_hod_models
from .. import hod_factory
from .. import mock_factory
from ...sim_manager import read_nbody

"""
def test_kravtsov04_mock():
	default_sim = read_nbody.processed_snapshot()
	kravtsov_model_dict = preloaded_hod_models.Kravtsov04()
	kravtsov_model = hod_factory.HodModel(kravtsov_model_dict)

	mock = mock_factory.HodMockFactory(
		default_sim, kravtsov_model, 
		populate=False)
"""