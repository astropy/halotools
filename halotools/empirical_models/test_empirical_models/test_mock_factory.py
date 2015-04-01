#!/usr/bin/env python

import numpy as np 
from .. import preloaded_models
from .. import hod_factory
from .. import mock_factory
from .. import preloaded_models
from ...sim_manager.generate_random_sim import FakeSim


def test_kravtsov04_mock():
	k = preloaded_models.Kravtsov04(threshold = -20)
	sim = FakeSim()
	mock = mock_factory.HodMockFactory(sim, k)
