#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest

import numpy as np 
from copy import copy 

from ....sim_manager import FakeSim
from ..prebuilt_model_factory import PrebuiltHodModelFactory
from ....custom_exceptions import HalotoolsError

__all__ = ['TestHodMockFactory']

class TestHodMockFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.HodMockFactory`. 
    """
    
    @pytest.mark.slow
    def test_mock_population_mask(self):

    	model = PrebuiltHodModelFactory('zheng07')

    	f100x = lambda t: t['halo_x'] > 100
    	f150z = lambda t: t['halo_z'] > 150

    	model.populate_mock(simname = 'fake', masking_function = f100x)
    	assert np.all(model.mock.galaxy_table['halo_x'] > 100)
    	model.populate_mock(simname = 'fake')
    	assert np.any(model.mock.galaxy_table['halo_x'] < 100)
    	model.populate_mock(simname = 'fake', masking_function = f100x)
    	assert np.all(model.mock.galaxy_table['halo_x'] > 100)

    	model.populate_mock(simname = 'fake', masking_function = f150z)
    	assert np.all(model.mock.galaxy_table['halo_z'] > 150)
    	assert np.any(model.mock.galaxy_table['halo_x'] < 100)
    	model.populate_mock(simname = 'fake')
    	assert np.any(model.mock.galaxy_table['halo_z'] < 150)




