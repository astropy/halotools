#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest
from astropy.config.paths import _find_home 

import numpy as np 
from copy import copy 

from ....sim_manager import FakeSim
from ..prebuilt_model_factory import PrebuiltHodModelFactory
from ....custom_exceptions import HalotoolsError

aph_home = u'/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

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

    @pytest.mark.xfail
    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_mock_population_pbcs(self):
        model = PrebuiltHodModelFactory('zheng07')
        model.populate_mock(simname = 'bolshoi', _testing_mode = True)
        assert model.mock._testing_mode == True

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_censat_positions1(self):
        model = PrebuiltHodModelFactory('zheng07')
        model.populate_mock(simname = 'bolshoi')

        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])

        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

        f100x = lambda t: t['halo_x'] > 100
        model.populate_mock(simname = 'bolshoi', masking_function = f100x)
        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

    @pytest.mark.slow
    def test_censat_positions2(self):
        model = PrebuiltHodModelFactory('zheng07')
        model.populate_mock(simname = 'fake')
        
        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

        f100x = lambda t: t['halo_x'] > 100
        model.populate_mock(simname = 'fake', masking_function = f100x)
        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])






