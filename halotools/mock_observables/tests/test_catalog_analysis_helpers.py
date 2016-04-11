#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from copy import deepcopy 

from collections import Counter

import numpy as np 

from astropy.tests.helper import pytest
from astropy.table import Table 

from .. import catalog_analysis_helpers as cat_helpers

from ...sim_manager import FakeSim

from ...custom_exceptions import HalotoolsError

__all__ = ['TestCatalogAnalysisHelpers']

class TestCatalogAnalysisHelpers(TestCase):
    """ Class providing tests of the `~halotools.mock_observables.catalog_analysis_helpers`. 
    """
    
    def setUp(self):

        halocat = FakeSim()
        self.halo_table = halocat.halo_table
        self.Lbox = halocat.Lbox

    def test_mean_y_vs_x1(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'])

    def test_mean_y_vs_x2(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'], 
            error_estimator = 'variance')

    def test_mean_y_vs_x3(self):
        with pytest.raises(HalotoolsError) as err:
            abscissa, mean, err = cat_helpers.mean_y_vs_x(
                self.halo_table['halo_mvir'], self.halo_table['halo_spin'], 
                error_estimator = 'Jose Canseco')
        substr = "Input ``error_estimator`` must be either"
        assert substr in err.value.args[0]

    def test_return_xyz_formatted_array1(self):
        x, y, z = (self.halo_table['halo_x'], 
            self.halo_table['halo_y'], self.halo_table['halo_z'])
        pos = cat_helpers.return_xyz_formatted_array(x, y, z)
        assert np.shape(pos) == (len(x), 3)

        mask = self.halo_table['halo_mvir'] >= 10**13.5
        masked_pos = cat_helpers.return_xyz_formatted_array(x, y, z, mask=mask)
        npts = len(self.halo_table[mask])
        assert np.shape(masked_pos) == (npts, 3)

        assert masked_pos.shape[0] < pos.shape[0]

        pos_zdist = cat_helpers.return_xyz_formatted_array(
            x, y, z, velocity = self.halo_table['halo_vz'], 
            velocity_distortion_dimension = 'z')
        assert np.all(pos_zdist[:,0] == pos[:,0])
        assert np.all(pos_zdist[:,1] == pos[:,1])
        assert np.any(pos_zdist[:,2] != pos[:,2])
        assert np.all(abs(pos_zdist[:,2] - pos[:,2]) < 50)

        pos_zdist_pbc = cat_helpers.return_xyz_formatted_array(
            x, y, z, velocity = self.halo_table['halo_vz'], 
            velocity_distortion_dimension = 'z', 
            period = self.Lbox)
        assert np.all(pos_zdist_pbc[:,0] == pos[:,0])
        assert np.all(pos_zdist_pbc[:,1] == pos[:,1])
        assert np.any(pos_zdist_pbc[:,2] != pos[:,2])

        assert np.any(abs(pos_zdist_pbc[:,2] - pos[:,2]) > 50)

    def tearDown(self):
        del self.halo_table









