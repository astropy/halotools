#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest
from astropy.config.paths import _find_home

import numpy as np
from copy import deepcopy

from ....mock_observables import return_xyz_formatted_array, tpcf_one_two_halo_decomp

from ....sim_manager import FakeSim, CachedHaloCatalog
from ....sim_manager.fake_sim import FakeSimHalosNearBoundaries
from ..prebuilt_model_factory import PrebuiltHodModelFactory
from ....custom_exceptions import HalotoolsError

aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ['TestHodMockFactory']


class TestHodMockFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.HodMockFactory`.
    """

    def setUp(self):
        self.model = PrebuiltHodModelFactory('zheng07', threshold=-21)
        self.fakesim = FakeSimHalosNearBoundaries()

        self.model.populate_mock(self.fakesim)

        self.galaxy_table1 = deepcopy(self.model.mock.galaxy_table)
        f100x = lambda t: t['halo_x'] > 100
        self.model.mock.populate(masking_function=f100x)
        self.galaxy_table2 = deepcopy(self.model.mock.galaxy_table)

    @pytest.mark.slow
    def test_mock_population_mask(self):

        model = PrebuiltHodModelFactory('zheng07')

        f100x = lambda t: t['halo_x'] > 100
        f150z = lambda t: t['halo_z'] > 150

        halocat = FakeSim()
        model.populate_mock(halocat, masking_function=f100x)
        assert np.all(model.mock.galaxy_table['halo_x'] > 100)
        model.populate_mock(halocat)
        assert np.any(model.mock.galaxy_table['halo_x'] < 100)
        model.populate_mock(halocat, masking_function=f100x)
        assert np.all(model.mock.galaxy_table['halo_x'] > 100)

        model.populate_mock(halocat, masking_function=f150z)
        assert np.all(model.mock.galaxy_table['halo_z'] > 150)
        assert np.any(model.mock.galaxy_table['halo_x'] < 100)
        model.populate_mock(halocat)
        assert np.any(model.mock.galaxy_table['halo_z'] < 150)

    def test_mock_population_pbcs(self):

        cenmask = self.galaxy_table1['gal_type'] == 'centrals'
        cens = self.galaxy_table1[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])

        sats = self.galaxy_table1[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

        cenmask = self.galaxy_table2['gal_type'] == 'centrals'
        cens = self.galaxy_table2[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = self.galaxy_table2[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

    @pytest.mark.slow
    def test_censat_positions2(self):

        model = PrebuiltHodModelFactory('zheng07')
        halocat = FakeSim()
        model.populate_mock(halocat)

        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

        f100x = lambda t: t['halo_x'] > 100
        model.populate_mock(halocat, masking_function=f100x)
        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        assert np.all(cens['halo_x'] == cens['x'])
        sats = model.mock.galaxy_table[~cenmask]
        assert np.any(sats['halo_x'] != sats['x'])

    @pytest.mark.slow
    def test_nonPBC_positions(self):

        model = PrebuiltHodModelFactory('zheng07', threshold=-18)

        halocat = FakeSimHalosNearBoundaries()
        model.populate_mock(halocat, enforce_PBC=False)

        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        sats = model.mock.galaxy_table[~cenmask]

        sats_outside_boundary_mask = (
            (sats['x'] < 0) | (sats['x'] > halocat.Lbox)
            | (sats['y'] < 0) | (sats['y'] > halocat.Lbox)
            | (sats['z'] < 0) | (sats['z'] > halocat.Lbox))
        assert np.any(sats_outside_boundary_mask == True)

        cens_outside_boundary_mask = (
            (cens['x'] < 0) | (cens['x'] > halocat.Lbox)
            | (cens['y'] < 0) | (cens['y'] > halocat.Lbox)
            | (cens['z'] < 0) | (cens['z'] > halocat.Lbox))
        assert np.all(cens_outside_boundary_mask == False)

    @pytest.mark.slow
    def test_PBC_positions(self):

        model = PrebuiltHodModelFactory('zheng07', threshold=-18)

        halocat = FakeSimHalosNearBoundaries()
        model.populate_mock(halocat=halocat, enforce_PBC=True,
            _testing_mode=True)

        cenmask = model.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model.mock.galaxy_table[cenmask]
        sats = model.mock.galaxy_table[~cenmask]

        sats_outside_boundary_mask = (
            (sats['x'] < 0) | (sats['x'] > halocat.Lbox)
            | (sats['y'] < 0) | (sats['y'] > halocat.Lbox)
            | (sats['z'] < 0) | (sats['z'] > halocat.Lbox))
        assert np.all(sats_outside_boundary_mask == False)

        cens_outside_boundary_mask = (
            (cens['x'] < 0) | (cens['x'] > halocat.Lbox)
            | (cens['y'] < 0) | (cens['y'] > halocat.Lbox)
            | (cens['z'] < 0) | (cens['z'] > halocat.Lbox))
        assert np.all(cens_outside_boundary_mask == False)

    def test_zero_satellite_edge_case(self):

        model = PrebuiltHodModelFactory('zheng07', threshold=-18)
        model.param_dict['logM0'] = 20

        halocat = FakeSim()
        model.populate_mock(halocat=halocat)

    def test_zero_halo_edge_case(self):

        model = PrebuiltHodModelFactory('zheng07', threshold=-18)
        model.param_dict['logM0'] = 20

        halocat = FakeSim()
        with pytest.raises(HalotoolsError) as err:
            model.populate_mock(halocat=halocat, Num_ptcl_requirement=1e10)
        substr = "Such a cut is not permissible."
        assert substr in err.value.args[0]

    @pytest.mark.slow
    def test_satellite_positions1(self):

        gals = self.galaxy_table1
        x1 = gals['x']
        y1 = gals['y']
        z1 = gals['z']
        x2 = gals['halo_x']
        y2 = gals['halo_y']
        z2 = gals['halo_z']
        dx = np.fabs(x1 - x2)
        dx = np.fmin(dx, self.model.mock.Lbox - dx)
        dy = np.fabs(y1 - y2)
        dy = np.fmin(dy, self.model.mock.Lbox - dy)
        dz = np.fabs(z1 - z2)
        dz = np.fmin(dz, self.model.mock.Lbox - dz)
        d = np.sqrt(dx*dx+dy*dy+dz*dz)
        assert np.all(d <= gals['halo_rvir'])

    @pytest.mark.slow
    @pytest.mark.skipif('not APH_MACHINE')
    def test_one_two_halo_decomposition_on_mock(self):
        """ Enforce that the one-halo term is exactly zero
        on sufficiently large scales.
        """
        model = PrebuiltHodModelFactory('zheng07', threshold=-21)
        bolshoi_halocat = CachedHaloCatalog(simname='bolshoi')
        model.populate_mock(bolshoi_halocat)
        gals = model.mock.galaxy_table
        pos = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'])
        halo_hostid = gals['halo_id']

        rbins = np.logspace(-1, 1.5, 15)
        xi_1h, xi_2h = tpcf_one_two_halo_decomp(pos, halo_hostid, rbins,
            period=model.mock.Lbox, num_threads='max')
        assert xi_1h[-1] == -1

        del model

    def tearDown(self):
        del self.model
        del self.galaxy_table1
        del self.galaxy_table2
